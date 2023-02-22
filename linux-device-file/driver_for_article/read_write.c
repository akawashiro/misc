/ *read_write.c * /
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/uaccess.h>

    MODULE_LICENSE("GPL");

#define DRIVER_MAJOR 333
#define DRIVER_NAME "read_write_driver"

static ssize_t driver_read(struct file *File, char *user_buffer, size_t count,
                           loff_t *offs) {
  user_buffer[0] = 'A';
  return 1;
}

static ssize_t driver_write(struct file *File, const char *user_buffer,
                            size_t count, loff_t *offs) {
  return 1;
}

static int driver_open(struct inode *device_file, struct file *instance) {
  printk("read_write_driver - open was called!\n");
  return 0;
}

static int driver_close(struct inode *device_file, struct file *instance) {
  printk("read_write_driver - close was called!\n");
  return 0;
}

static struct file_operations fops = {.open = driver_open,
                                      .release = driver_close,
                                      .read = driver_read,
                                      .write = driver_write};

static int __init ModuleInit(void) {
  printk("read_write_driver - ModuleInit was called!\n");
  register_chrdev(DRIVER_MAJOR, DRIVER_NAME, &fops);
  return 0;
}

static void __exit ModuleExit(void) {
  printk("read_write_driver - ModuleExit was called!\n");
  unregister_chrdev(DRIVER_MAJOR, DRIVER_NAME);
}

module_init(ModuleInit);
module_exit(ModuleExit);