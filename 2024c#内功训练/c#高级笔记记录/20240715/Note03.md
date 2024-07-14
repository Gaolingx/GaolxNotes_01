# C#高级编程之——反射（三）

## 三、反射入门——反射的使用（案例篇）

在上期教程中，我们讲解了Type类型的概念，简单使用了typeof 运算符以及GetType 方法通过实例对象获取类型，接下来我们继续深入讲解Type类的使用。

### 详细知识点

1. 获取属性

通过对"Console.WriteLine($"StudentInfo类型中属性的名称:{propertyInfo.Name},类型:{propertyInfo.PropertyType}");"进行断点，观察PropertyInfo中各字段的信息。