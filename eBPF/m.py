from bcc import BPF

program = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>

static void handle_open(struct pt_regs *ctx, const char *filename)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    bpf_trace_printk("openat: filename=%s pid=%d\\n", filename, pid);
}

int syscall__open(struct pt_regs *ctx,
    const char __user *filename,
    int flags,
    umode_t mode)
{
    handle_open(ctx, filename);
    return 0;
}

int syscall__openat(struct pt_regs *ctx,
    int dfd,
    const char __user *filename,
    int flags,
    umode_t mode)
{
    handle_open(ctx, filename);
    return 0;
}

int syscall__ioctl(struct pt_regs *ctx,
    unsigned int fd,
    unsigned int cmd,
    unsigned long arg)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    bpf_trace_printk("ioctl: fd=%d cmd=%d pid=%d\\n", fd, cmd, pid);
    return 0;
}
"""

b = BPF(text=program)
syscall = b.get_syscall_fnname("open")
b.attach_kprobe(event=syscall, fn_name="syscall__open")
syscall = b.get_syscall_fnname("openat")
b.attach_kprobe(event=syscall, fn_name="syscall__openat")
syscall = b.get_syscall_fnname("ioctl")
b.attach_kprobe(event=syscall, fn_name="syscall__ioctl")

print("Tracing... Ctrl-C to end.")
b.trace_print()
