// Android host layer (runtime/platform.h), arm64. The Linux layer's approach (one read/write guest window, sigaction
// crash handler), with what Android does differently: the app is forked from the zygote (no re-exec, no fresh address
// space: the guest window sits GUEST_SHIFT lower, runtime/cpu.h), there is no terminal (stderr goes to logcat and to
// lasrecomp.log in the data folder), backtraces come from the unwinder, and the "executable" is libmain.so.
#define _GNU_SOURCE
#include <SDL3/SDL.h>
#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/ucontext.h>
#include <time.h>
#include <unistd.h>
#include <unwind.h>

#include "../platform.h"
#include "android.h"

void rt_crash_explain(char *out, size_t n, const char *log_path);  // runtime/crash_explain.c

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

// ---- guest address space
#include "../rt.h"
// As on Linux: everything readable/writable from the start, untouched pages cost nothing (MAP_NORESERVE).
bool plat_reserve_guest(uint64_t lo, uint64_t hi) {
    void *p = mmap((void *)(uintptr_t)lo, hi - lo, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return false;
    if (p != (void *)(uintptr_t)lo) { munmap(p, hi - lo); return false; }  // (kernels before 4.17: just a hint)
    return true;
}

static void wt_forget(uint64_t lo, uint64_t hi);
bool plat_reset_guest(uint64_t lo, uint64_t hi) {
    bool ok = mmap((void *)(uintptr_t)lo, hi - lo, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED,
                   -1, 0) == (void *)(uintptr_t)lo;
    wt_forget(lo, hi);  // (fresh writable pages: none of them clean)
    return ok;
}


// Page write tracking for the guest heap (the renderer's vertex cache and texture checks: plat_written_pages), by
// write protection: a query reports the range's pages written since the last one (or never asked about), marks them
// clean and makes the range read-only; the first write to a page then faults, and the handler marks it written and
// gives the page its write access back. The OS writing into guest memory (file reads) comes through plat_commit first.
// LARECOMP_WRITE_TRACKING=0: none (every range counts as written).
enum { WT_UNTRACKED = 0, WT_CLEAN = 1, WT_WRITTEN = 2 };
static _Atomic uint8_t *wt_state;  // per heap page
static _Atomic int wt_on = -1;      // (set once wt_state exists: other threads test it before touching the table)
#define WT_PAGE 4096ull

// A heap fault write tracking couldn't resolve (the page stayed read-only: an mprotect the system refused) left this
// marker in the data folder (crash_handler): the runs after it go without tracking - every range then counts as
// written, a little slower but nothing protected. (Reports: Poco X3 Pro, crashes in the game's allocator mutex at
// HEAP_BASE+0x48 on starting the game.) LARECOMP_WRITE_TRACKING=1 turns it on again regardless.
static char g_wt_marker[1100];

static void wt_setup(void) {
    const char *e = getenv("LARECOMP_WRITE_TRACKING");
    bool on = (!e || atoi(e)) && sysconf(_SC_PAGESIZE) == (long)WT_PAGE;
    if (on && !e && g_wt_marker[0] && access(g_wt_marker, F_OK) == 0) {
        on = false;
        fprintf(stderr, "[mem] heap write tracking off (a heap fault in an earlier run)\n");
    }
    if (on) wt_state = calloc(HEAP_MAX / WT_PAGE, 1);
    atomic_store(&wt_on, on && wt_state ? 1 : 0);
}

static bool wt_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    if (atomic_load(&wt_on) < 0) pthread_once(&once, wt_setup);
    return atomic_load(&wt_on) > 0;
}

static void wt_forget(uint64_t lo, uint64_t hi) {
    if (atomic_load(&wt_on) <= 0 || hi <= HEAP_BASE || lo >= HEAP_BASE + HEAP_MAX) return;
    uint64_t a = lo > HEAP_BASE ? (lo - HEAP_BASE) / WT_PAGE : 0, b = (hi - HEAP_BASE + WT_PAGE - 1) / WT_PAGE;
    if (b > HEAP_MAX / WT_PAGE) b = HEAP_MAX / WT_PAGE;
    for (uint64_t p = a; p < b; p++) atomic_store_explicit(&wt_state[p], WT_UNTRACKED, memory_order_relaxed);
}

static bool wt_range(uint64_t addr, uint64_t size, uint64_t *p0, uint64_t *p1) {
    if (!size || addr < HEAP_BASE || addr - HEAP_BASE >= HEAP_MAX) return false;
    uint64_t end = addr + size - HEAP_BASE;
    if (end > HEAP_MAX) end = HEAP_MAX;
    *p0 = (addr - HEAP_BASE) / WT_PAGE, *p1 = (end + WT_PAGE - 1) / WT_PAGE;
    return true;
}

int plat_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, int max, uint64_t *page) {
    *page = WT_PAGE;
    uint64_t p0, p1;
    if (!wt_init() || !wt_range(addr, size, &p0, &p1)) return -1;
    // Clean pages are protected already: only the runs of written (or never asked about) pages are marked clean and
    // protected again - clean first, then protected: a write in between lands before the copy. (The whole range each
    // time cost an mprotect per texture check and vertex buffer, and a TLB flush with it.)
    int n = 0;
    uint64_t run = p1;  // (start of the current run of pages to protect)
    for (uint64_t p = p0; p <= p1; p++) {
        if (p < p1 && atomic_load_explicit(&wt_state[p], memory_order_relaxed) != WT_CLEAN &&
            atomic_exchange(&wt_state[p], WT_CLEAN) != WT_CLEAN) {
            if (n < max) pages[n] = HEAP_BASE + p * WT_PAGE;
            n++;
            if (run == p1) run = p;
            continue;
        }
        if (run != p1) mprotect((void *)(uintptr_t)(HEAP_BASE + run * WT_PAGE), (p - run) * WT_PAGE, PROT_READ), run = p1;
    }
    return n > max ? max : n;
}

// A write fault in the heap (the only faults there are ours): page writable again, then marked written - in that
// order: marked first, a query in between could mark it clean and protect it before our mprotect opened it again,
// leaving a writable page that counts as clean (its writes never seen). This way such a query either reports the page
// (and the write faults once more after its protect) or finds it clean with our mark still to come. Signal handler.
static bool wt_fault(uint64_t fa) {
    if (atomic_load(&wt_on) <= 0 || fa < HEAP_BASE || fa - HEAP_BASE >= HEAP_MAX) return false;
    uint64_t p = (fa - HEAP_BASE) / WT_PAGE;
    void *pg = (void *)(uintptr_t)(HEAP_BASE + p * WT_PAGE);
    // (an mprotect can fail for a moment - EAGAIN, or ENOMEM with the mapping split into too many pieces: tried
    // again, then the page's whole 64 KB neighbourhood, which merges pieces back)
    int r = -1;
    for (int i = 0; i < 3 && r != 0; i++) r = mprotect(pg, WT_PAGE, PROT_READ | PROT_WRITE);
    if (r != 0) {
        uint64_t lo = (fa - HEAP_BASE) & ~0xFFFFull;
        r = mprotect((void *)(uintptr_t)(HEAP_BASE + lo), 0x10000, PROT_READ | PROT_WRITE);
        if (r == 0)
            for (uint64_t q = lo / WT_PAGE; q < (lo + 0x10000) / WT_PAGE; q++) atomic_store(&wt_state[q], WT_WRITTEN);
    }
    if (r != 0) {
        char m[96];
        int n = snprintf(m, sizeof m, "[mem] write tracking: mprotect %p failed, errno %d\n", pg, errno);
        write(2, m, n > 0 ? (size_t)n : 0);
        return false;
    }
    atomic_store(&wt_state[p], WT_WRITTEN);
    return true;
}

// (the OS is about to write [addr, addr + size): a read-only page would make the call fail rather than fault)
void plat_commit(uint64_t addr, uint64_t size) {
    uint64_t p0, p1;
    if (atomic_load(&wt_on) <= 0 || !wt_range(addr, size, &p0, &p1)) return;
    // writable first, then marked (as wt_fault); always (file reads: rare) - a page seen written here could be
    // marked clean and protected by a query in between
    mprotect((void *)(uintptr_t)(HEAP_BASE + p0 * WT_PAGE), (p1 - p0) * WT_PAGE, PROT_READ | PROT_WRITE);
    for (uint64_t p = p0; p < p1; p++) atomic_store(&wt_state[p], WT_WRITTEN);
}

int plat_protect(uint64_t addr, uint64_t size, int prot) {
    int p = prot == PLAT_PROT_NONE ? PROT_NONE : prot == PLAT_PROT_READ ? PROT_READ : PROT_READ | PROT_WRITE;
    return mprotect((void *)(uintptr_t)addr, size, p);
}

// ---- time
uint64_t plat_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

void plat_sleep_until_ns(uint64_t t) {
    struct timespec ts = {.tv_sec = (time_t)(t / 1000000000u), .tv_nsec = (long)(t % 1000000000u)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
}

// Apps can't have SCHED_FIFO; the most urgent nice value an app may set (Android's display / audio threads use it).
void plat_thread_realtime(uint64_t period_ns) {
    (void)period_ns;
    setpriority(PRIO_PROCESS, (id_t)gettid(), -10);
}

// ---- log
// stdout / stderr into a pipe; a thread hands each line to logcat (tag LASRecomp) and appends it to lasrecomp.log in
// the data folder (the previous run's log is kept as lasrecomp.prev.log: after a crash, the app is started again).
static char g_log_path[1024];
static FILE *g_log_file;

static void *log_pump(void *arg) {
    plat_thread_name("log");
    int fd = (int)(intptr_t)arg;
    char buf[4096], line[4096];
    size_t n_line = 0;
    for (;;) {
        ssize_t n = read(fd, buf, sizeof buf);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        if (g_log_file) { fwrite(buf, 1, (size_t)n, g_log_file); fflush(g_log_file); }
        for (ssize_t i = 0; i < n; i++) {
            if (buf[i] == '\n' || n_line == sizeof line - 1) {
                line[n_line] = 0;
                __android_log_write(ANDROID_LOG_INFO, "LASRecomp", line);
                n_line = 0;
                if (buf[i] == '\n') continue;
            }
            line[n_line++] = buf[i];
        }
    }
    return NULL;
}

static void log_start(void) {
    const char *dir = SDL_GetAndroidExternalStoragePath();
    if (!dir) dir = SDL_GetAndroidInternalStoragePath();
    if (dir) {
        char prev[1100];
        snprintf(g_log_path, sizeof g_log_path, "%s/lasrecomp.log", dir);
        snprintf(g_wt_marker, sizeof g_wt_marker, "%s/.no_write_tracking", dir);
        snprintf(prev, sizeof prev, "%s/lasrecomp.prev.log", dir);
        rename(g_log_path, prev);
        g_log_file = fopen(g_log_path, "w");
    }
    int p[2];
    if (pipe(p) != 0) return;
    dup2(p[1], STDERR_FILENO);
    dup2(p[1], STDOUT_FILENO);
    close(p[1]);
    pthread_t t;
    pthread_create(&t, NULL, log_pump, (void *)(intptr_t)p[0]);
    pthread_detach(t);
}

void plat_init(void) { log_start(); }

// The "executable": libmain.so (SDL's convention), possibly inside the APK (".../base.apk!/lib/arm64-v8a/libmain.so":
// libraries stored uncompressed are loaded from the APK in place).
bool plat_exe_path(char *buf, size_t size) {
    Dl_info di = {0};
    if (!dladdr((void *)plat_exe_path, &di) || !di.dli_fname) return false;
    snprintf(buf, size, "%s", di.dli_fname);
    return true;
}

void plat_exec(char *const argv[]) { (void)argv; }  // (an app can't replace its process; callers handle the return)

// ---- diagnostics
const char *plat_symbol(uint64_t pc, uint64_t *offset) {
    Dl_info di = {0};
    if (!dladdr((void *)(uintptr_t)pc, &di) || !di.dli_sname) return NULL;
    if (offset) *offset = pc - (uint64_t)(uintptr_t)di.dli_saddr;
    return di.dli_sname;
}

void plat_fatal_notice(void) {
    fflush(stderr);
    char text[1600];
    rt_crash_explain(text, sizeof text, g_log_path[0] ? g_log_path : "logcat (tag LASRecomp)");
    // (a graphics driver the game can't run on: the player can install one right here - the settings that have that
    // row are out of reach when the driver fails at the start screen)
    // (any case: the headline is "Graphics driver error" - matched as "graphics driver" only, it never showed)
    if (strcasestr(text, "graphics driver") && android_driver_error_dialog(text)) return;
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "LAS Recomp", text, NULL);
}

bool plat_module_name(uint64_t pc, char *out, size_t n) {
    Dl_info di = {0};
    if (!dladdr((void *)(uintptr_t)pc, &di) || !di.dli_fname) return false;
    snprintf(out, n, "%s", di.dli_fname);
    return true;
}

const char *rt_host_fn_name(uint64_t pc, uint64_t *offset);  // runtime/dispatch.c

typedef struct { void **pcs; int n, max; } Frames;
static _Unwind_Reason_Code unwind_cb(struct _Unwind_Context *ctx, void *arg) {
    Frames *f = arg;
    uintptr_t pc = _Unwind_GetIP(ctx);
    if (pc && f->n < f->max) f->pcs[f->n++] = (void *)pc;
    return f->n < f->max ? _URC_NO_REASON : _URC_END_OF_STACK;
}

void plat_print_backtrace(void) {
    void *bt[64];
    Frames fr = {bt, 0, 64};
    _Unwind_Backtrace(unwind_cb, &fr);
    Dl_info self = {0};
    dladdr((void *)plat_print_backtrace, &self);
    for (int i = 0; i < fr.n; i++) {
        uint64_t pc = (uint64_t)(uintptr_t)bt[i], off = 0;
        Dl_info di = {0};
        if (!dladdr(bt[i], &di)) di.dli_fname = NULL;
        bool ours = di.dli_fbase && di.dli_fbase == self.dli_fbase;
        const char *s = plat_symbol(pc, &off);
        if (!s && ours) s = rt_host_fn_name(pc, &off);
        const char *mod = di.dli_fname ? (strrchr(di.dli_fname, '/') ? strrchr(di.dli_fname, '/') + 1 : di.dli_fname) : "?";
        uint64_t moff = pc - (uint64_t)(uintptr_t)di.dli_fbase;
        if (s) fprintf(stderr, "  #%-2d %016llx %s+%llx (%s+%llx)\n", i, (unsigned long long)pc, s, (unsigned long long)off,
                       mod, (unsigned long long)moff);
        else fprintf(stderr, "  #%-2d %016llx %s+%llx\n", i, (unsigned long long)pc, mod, (unsigned long long)moff);
    }
}

static void crash_handler(int sig, siginfo_t *si, void *ctx) {
    ucontext_t *uc = ctx;
    static volatile int depth;
    if (sig == SIGABRT && depth++ > 0) { signal(SIGABRT, SIG_DFL); abort(); }  // (nested: just die)
    uint64_t fa = (uint64_t)(uintptr_t)si->si_addr;
    uint64_t pc = uc->uc_mcontext.pc, lr = uc->uc_mcontext.regs[30];
    // The write tracking protects the heap pages read-only while the guest writes to them, and the kernel can then
    // report the fault as SEGV_MAPERR (the address is not mapped) even though the page stays mapped and resident: an
    // mprotect splitting the VMA races with the guest write, and the access resolves against the VMA being replaced.
    // Measured on the device: 20/20 MAPERR with a concurrent mprotect, 0/20 without one. Both codes mean the same
    // thing here (the page needs its write access back), and wt_fault() fails harmlessly on a page that is really
    // unmapped (mprotect returns ENOMEM, and so does the 64 KB neighbourhood attempt), so a real hole still falls
    // through to the diagnostic path below.
    if (sig == SIGSEGV && (si->si_code == SEGV_ACCERR || si->si_code == SEGV_MAPERR) && wt_fault(fa)) return;
    if (rt_fault_watch(fa, pc)) return;
    if (sig == SIGSEGV && fa >= HEAP_BASE && fa - HEAP_BASE < HEAP_MAX) {  // (see wt_setup)
        char m[96];
        int n = snprintf(m, sizeof m, "[mem] heap fault, si_code %d, tracking %d\n", si->si_code, (int)atomic_load(&wt_on));
        write(2, m, n > 0 ? (size_t)n : 0);
        if (atomic_load(&wt_on) > 0 && g_wt_marker[0]) {
            int fd = open(g_wt_marker, O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (fd >= 0) close(fd);
        }
    }
    {   // the crashing thread and its callers by frame pointers from the signal's context (the unwinder can't get past
        // the signal frame into code without unwind tables, e.g. a GPU driver): module + offset, for crash reports
        char m[200], name[32] = "?";
        prctl(PR_GET_NAME, name, 0, 0, 0);
        int n = snprintf(m, sizeof m, "[crash] thread \"%s\", frames from the fault:\n", name);
        write(2, m, n > 0 ? (size_t)n : 0);
        uint64_t fp = uc->uc_mcontext.regs[29], f[18] = {pc, lr};
        int nf = 2;
        for (int i = 0; i < 16 && fp && !(fp & 15) && fp > 0x10000; i++) {
            uint64_t ret = ((uint64_t *)(uintptr_t)fp)[1], next = ((uint64_t *)(uintptr_t)fp)[0];
            if (!ret) break;
            f[nf++] = ret;
            if (next <= fp) break;
            fp = next;
        }
        for (int i = 0; i < nf; i++) {
            Dl_info di = {0};
            bool ok = dladdr((void *)(uintptr_t)f[i], &di) && di.dli_fname;
            const char *mod = ok ? (strrchr(di.dli_fname, '/') ? strrchr(di.dli_fname, '/') + 1 : di.dli_fname) : "?";
            n = snprintf(m, sizeof m, "  %016llx %s+%llx%s%s\n", (unsigned long long)f[i], mod,
                         (unsigned long long)(ok ? f[i] - (uint64_t)(uintptr_t)di.dli_fbase : 0), di.dli_sname ? " " : "",
                         di.dli_sname ? di.dli_sname : "");
            write(2, m, n > 0 ? (size_t)n : 0);
        }
    }
    char what[32];
    snprintf(what, sizeof what, "signal %d", sig);
    rt_fault_report(what, fa, pc, lr);  // (ends in abort(): the system's tombstone follows)
}

void plat_crash_handler_install(void) {
    stack_t ss = {.ss_sp = malloc(1 << 20), .ss_size = 1 << 20};
    sigaltstack(&ss, NULL);
    struct sigaction sa = {0};
    sa.sa_sigaction = crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}

// ---- child processes: as on Linux (posix_spawn). Android runs programs only from the app's native library folder
// (lib*.so in jniLibs, extracted at install), e.g. the on-phone game code build's python and compiler
// (docs/android-ondevice-build.md); nothing starts any yet.
#include <ftw.h>
#include <spawn.h>
#include <sys/wait.h>
extern char **environ;

PlatProc plat_spawn(const char *const argv[], const char *log_path, bool append_log) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log_path ? log_path : "/dev/null",
                                     O_WRONLY | O_CREAT | (append_log ? O_APPEND : O_TRUNC), 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    posix_spawnattr_t at;  // (its own process group: plat_proc_kill ends what it starts)
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at, 0);
    pid_t pid;
    int rc = posix_spawn(&pid, argv[0], &fa, &at, (char *const *)argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    return rc ? 0 : (PlatProc)pid;
}

void plat_proc_kill(PlatProc p) {
    kill(-(pid_t)p, SIGKILL);
    waitpid((pid_t)p, NULL, 0);
}

int plat_proc_wait(PlatProc p, bool block) {
    int st = 0;
    pid_t r = waitpid((pid_t)p, &st, block ? 0 : WNOHANG);
    if (r == 0) return -1;
    if (r < 0) return 255;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 255;
}

static int rm_entry(const char *path, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb, (void)flag, (void)ftw;
    return remove(path);
}
bool plat_remove_tree(const char *dir) { return nftw(dir, rm_entry, 16, FTW_DEPTH | FTW_PHYS) == 0; }

void plat_terminate(void) { _exit(0); }

void plat_thread_name(const char *name) { pthread_setname_np(pthread_self(), name); }

// Phones have one or two fast cores (the S20 FE's Snapdragon 865: cpu7 at 2.84 GHz, cpu4-6 at 2.42, cpu0-3 small) and
// the game runs on one thread plus the render thread: those two get the big cores (see core_scan), the game's other
// threads anything but the fastest. Cores by their top frequency. LARECOMP_AFFINITY=0: leave it to Android.
static cpu_set_t g_core_sets[3];
static bool g_core_on;

static void core_scan(void) {
    long freq[CPU_SETSIZE] = {0}, best = 0, second = 0;
    int n = 0, top = -1, ntop = 0, nsecond = 0;
    for (; n < CPU_SETSIZE; n++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", n);
        FILE *f = fopen(path, "r");
        if (!f) break;
        if (fscanf(f, "%ld", &freq[n]) != 1) freq[n] = 0;
        fclose(f);
        if (freq[n] > best) best = freq[n], top = n;
    }
    for (int i = 0; i < n; i++) if (freq[i] < best && freq[i] > second) second = freq[i];
    for (int i = 0; i < n; i++) ntop += freq[i] == best, nsecond += freq[i] == second;
    const char *e = getenv("LARECOMP_AFFINITY");
    if ((e && !strcmp(e, "0")) || ntop != 1 || !nsecond) return;  // (no single fastest core: nothing to do)
    for (int r = 0; r < 3; r++) CPU_ZERO(&g_core_sets[r]);
    // The render thread gets the fastest core (it's the busier one: the driver's work as well as ours - an S20 FE's
    // dungeons got nearer 60 that way), the game's main thread the other big ones. LARECOMP_AFFINITY=main: the other
    // way round; =big: both on every big core, the scheduler choosing
    int mode = e && !strcmp(e, "main") ? 0 : e && !strcmp(e, "big") ? 2 : 1;
    for (int i = 0; i < n; i++) {
        bool fast = i == top, big = freq[i] == second;
        if (mode == 2 ? fast || big : mode == 1 ? big : fast) CPU_SET(i, &g_core_sets[PLAT_CORE_MAIN]);
        if (mode == 2 ? fast || big : mode == 1 ? fast : big) CPU_SET(i, &g_core_sets[PLAT_CORE_RENDER]);
        if (i != top) CPU_SET(i, &g_core_sets[PLAT_CORE_OTHER]);
    }
    g_core_on = true;
    fprintf(stderr, "[cpu] fastest core cpu%d (%ld MHz) for the %s thread, %d big cores (%ld MHz) for the %s\n", top,
            best / 1000, mode == 0 ? "game's" : mode == 1 ? "render" : "game's and render", nsecond, second / 1000,
            mode == 0 ? "render thread" : mode == 1 ? "game's" : "same");
}

// Performance hints (APerformanceHint, Android 13+, looked up at run time: the app starts on Android 11): one session
// for the game's main thread and the render thread, told each frame how long the game thread worked against the
// frame's target. That is what gets a phone's fastest core out of isolation and the clocks up (core_ctl parks the
// prime core while nothing it counts as heavy runs; a 70% thread on a big core isn't).
static pid_t g_hint_tids[2];
static void *g_hint_session;
static void *(*p_get_manager)(void);
static void *(*p_create_session)(void *, const int32_t *, size_t, int64_t);
static int (*p_update_target)(void *, int64_t);
static int (*p_report)(void *, int64_t);
static int (*p_set_threads)(void *, const pid_t *, size_t);

static void hint_load(void) {
    void *lib = dlopen("libandroid.so", RTLD_NOW);
    if (!lib) return;
    p_get_manager = (void *(*)(void))dlsym(lib, "APerformanceHint_getManager");
    p_create_session = (void *(*)(void *, const int32_t *, size_t, int64_t))dlsym(lib, "APerformanceHint_createSession");
    p_update_target = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_updateTargetWorkDuration");
    p_report = (int (*)(void *, int64_t))dlsym(lib, "APerformanceHint_reportActualWorkDuration");
    p_set_threads = (int (*)(void *, const pid_t *, size_t))dlsym(lib, "APerformanceHint_setThreads");
}

void plat_thread_core(int role) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, core_scan);
    if (role == PLAT_CORE_MAIN || role == PLAT_CORE_RENDER) {  // (the hint session's threads)
        pid_t tid = gettid(), *slot = &g_hint_tids[role == PLAT_CORE_RENDER];
        if (*slot != tid) {
            *slot = tid;
            if (g_hint_session && p_set_threads) {
                pid_t t[2];
                size_t n = 0;
                for (int i = 0; i < 2; i++) if (g_hint_tids[i]) t[n++] = g_hint_tids[i];
                p_set_threads(g_hint_session, t, n);
            }
        }
    }
    if (!g_core_on || role < 0 || role > 2) return;
    // the fastest core alone; while it's parked (isolated: EINVAL) the big ones
    if (sched_setaffinity(0, sizeof g_core_sets[role], &g_core_sets[role]) == 0) return;
    cpu_set_t big = g_core_sets[PLAT_CORE_RENDER];
    CPU_OR(&big, &big, &g_core_sets[PLAT_CORE_MAIN]);
    if (role == PLAT_CORE_OTHER || sched_setaffinity(0, sizeof big, &big) != 0)
        fprintf(stderr, "[cpu] affinity: %s\n", strerror(errno));
}

void plat_frame_work(uint64_t busy_ns, uint64_t target_ns) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    static int64_t target;
    static bool failed;
    const char *e = getenv("LARECOMP_PERF_HINT");
    if (failed || (e && !atoi(e))) return;
    pthread_once(&once, hint_load);
    if (!g_hint_session) {
        if (!p_get_manager || !p_create_session || !p_report || !g_hint_tids[0]) { failed = true; return; }
        void *m = p_get_manager();
        int32_t t[2];
        size_t n = 0;
        for (int i = 0; i < 2; i++) if (g_hint_tids[i]) t[n++] = g_hint_tids[i];
        g_hint_session = m ? p_create_session(m, t, n, (int64_t)target_ns) : NULL;
        if (!g_hint_session) {
            failed = true;
            fprintf(stderr, "[cpu] no performance hint session\n");
            return;
        }
        target = (int64_t)target_ns;
        fprintf(stderr, "[cpu] performance hint session: %zu threads, target %.1f ms\n", n, target_ns / 1e6);
    }
    if ((int64_t)target_ns != target && p_update_target) p_update_target(g_hint_session, target = (int64_t)target_ns);
    if (busy_ns) p_report(g_hint_session, (int64_t)busy_ns);
}

uint64_t plat_thread_cpu_ns(void) {  // (diagnostics: LARECOMP_LOG_PACE; pace.c's 30 fps frame cost)
    struct timespec ts;
    return clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0 ? (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec : 0;
}

void plat_thread_guest(int prio) { (void)prio; }  // (plat_thread_core places the threads on phones)

int plat_cpu_count(void) {
    cpu_set_t set;
    return sched_getaffinity(0, sizeof set, &set) == 0 ? CPU_COUNT(&set) : 0;
}

// (phones: in the log; the game keeps running)
void plat_notice(const char *text) { __android_log_print(ANDROID_LOG_WARN, "LASRecomp", "[notice] %s", text); }

// ---- the machine, for the log (runtime/main.c log_startup_info): the Android version, the phone and its SoC
#include <sys/system_properties.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
void plat_system_info(PlatSysInfo *si) {
    char rel[PROP_VALUE_MAX] = "", sdk[PROP_VALUE_MAX] = "", maker[PROP_VALUE_MAX] = "", model[PROP_VALUE_MAX] = "";
    char soc_m[PROP_VALUE_MAX] = "", soc[PROP_VALUE_MAX] = "", plat[PROP_VALUE_MAX] = "";
    __system_property_get("ro.build.version.release", rel);
    __system_property_get("ro.build.version.sdk", sdk);
    __system_property_get("ro.product.manufacturer", maker);
    __system_property_get("ro.product.model", model);
    __system_property_get("ro.soc.manufacturer", soc_m);
    __system_property_get("ro.soc.model", soc);
    __system_property_get("ro.board.platform", plat);
    struct utsname u;
    if (uname(&u) == 0) snprintf(si->os, sizeof si->os, "Android %s (API %s, %s, kernel %.32s)", rel, sdk, u.machine, u.release);
    else snprintf(si->os, sizeof si->os, "Android %s (API %s)", rel, sdk);
    snprintf(si->cpu, sizeof si->cpu, "%s %s, %s %s", maker, model, soc[0] ? soc_m : "", soc[0] ? soc : plat);
    si->threads = si->cores = (int)sysconf(_SC_NPROCESSORS_CONF);
    {   // big.LITTLE clusters by their top frequency: "1x2.84 + 3x2.42 + 4x1.80 GHz"
        long khz[16] = {0};
        int nk = 0, cnt[16] = {0};
        for (int c = 0; c < si->cores && c < 64; c++) {
            char p[96];
            snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
            FILE *f = fopen(p, "r");
            long v = 0;
            if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
            if (!v) continue;
            int i = 0;
            while (i < nk && khz[i] != v) i++;
            if (i == nk && nk < 16) khz[nk++] = v;
            if (i < 16) cnt[i]++;
        }
        int n = 0;
        for (int i = nk - 1; i >= 0 && n < (int)sizeof si->cores_detail - 16; i--)
            n += snprintf(si->cores_detail + n, sizeof si->cores_detail - n, "%s%dx%.2f", n ? " + " : "", cnt[i], khz[i] / 1e6);
        if (n) snprintf(si->cores_detail + n, sizeof si->cores_detail - n, " GHz");
    }
    struct sysinfo s;
    if (sysinfo(&s) == 0) si->ram = (uint64_t)s.totalram * s.mem_unit;
    si->package = "APK";
}

bool plat_onedrive_synced(const char *dir) { (void)dir; return false; }
