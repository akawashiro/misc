#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>

/* Meta Information */
MODULE_LICENSE("GPL");

#define DRIVER_MAJOR 333
#define DRIVER_NAME "read_write_driver"

static ssize_t driver_read(struct file *File, char *user_buffer, size_t count, loff_t *offs) {
    user_buffer[0] = 'A';
    return 1;
}

static ssize_t driver_write(struct file *File, const char *user_buffer, size_t count, loff_t *offs) {
    return 1;
}

static int driver_open(struct inode *device_file, struct file *instance) {
	printk("dev_nr - open was called!\n");
	return 0;
}

static int driver_close(struct inode *device_file, struct file *instance) {
	printk("dev_nr - close was called!\n");
	return 0;
}

static struct file_operations fops = {
	.owner = THIS_MODULE,
	.open = driver_open,
	.release = driver_close,
	.read = driver_read,
	.write = driver_write
};

static int __init ModuleInit(void) {
	printk("Hello, Kernel!\n");
    register_chrdev(DRIVER_MAJOR, DRIVER_NAME, &fops);
    return 0;
}

static void __exit ModuleExit(void) {
    printk("myDevice_exit\n");
    unregister_chrdev(DRIVER_MAJOR, DRIVER_NAME);
	printk("Goodbye, Kernel\n");
}

module_init(ModuleInit);
module_exit(ModuleExit);
