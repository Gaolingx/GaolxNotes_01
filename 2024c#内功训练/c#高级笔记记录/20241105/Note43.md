# C# 泛型集合（十一）自定义引用类型排序规则与相等规则

## 十一、泛型集合——拓展

在C#中，`HashSet<T>` 可以通过自定义相等性规则和哈希码规则来控制集合中元素的唯一性。为了自定义引用类型的排序规则，你通常需要使用其他数据结构，比如 `SortedSet<T>` 或 `List<T>` 结合排序方法。

### 1. 自定义相等性规则和哈希码规则

为了自定义相等性规则和哈希码规则，你需要实现 `IEquatable<T>` 接口，并重写 `GetHashCode` 和 `Equals` 方法。然后，你可以将这个类型传递给 `HashSet<T>` 的构造函数。

```csharp
public class MyCustomType : IEquatable<MyCustomType>
{
    public int Id { get; set; }
    public string Name { get; set; }

    public override int GetHashCode()
    {
        // 你可以根据实际需要组合多个字段来计算哈希码
        return Id.GetHashCode() ^ Name.GetHashCode();
    }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((MyCustomType)obj);
    }

    public bool Equals(MyCustomType other)
    {
        if (ReferenceEquals(null, other)) return false;
        if (ReferenceEquals(this, other)) return true;
        return Id == other.Id && Name == other.Name;
    }
}

// 使用自定义类型创建 HashSet
HashSet<MyCustomType> hashSet = new HashSet<MyCustomType>();
```

### 自定义排序规则

由于 `HashSet<T>` 不支持排序，你可以使用 `SortedSet<T>` 或 `List<T>` 结合 `List<T>.Sort` 方法来实现排序。为了使用 `SortedSet<T>`，你需要实现 `IComparable<T>` 接口。

```csharp
public class MyCustomType : IEquatable<MyCustomType>, IComparable<MyCustomType>
{
    public int Id { get; set; }
    public string Name { get; set; }

    public override int GetHashCode()
    {
        return Id.GetHashCode() ^ Name.GetHashCode();
    }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((MyCustomType)obj);
    }

    public bool Equals(MyCustomType other)
    {
        if (ReferenceEquals(null, other)) return false;
        if (ReferenceEquals(this, other)) return true;
        return Id == other.Id && Name == other.Name;
    }

    public int CompareTo(MyCustomType other)
    {
        if (other == null) return 1;
        int result = Id.CompareTo(other.Id);
        if (result != 0)
        {
            return result;
        }
        return Name.CompareTo(other.Name);
    }
}

// 使用 SortedSet 进行排序
SortedSet<MyCustomType> sortedSet = new SortedSet<MyCustomType>();
```

### 使用 List 和排序方法

如果你更喜欢使用 `List<T>`，你可以通过调用 `List<T>.Sort` 方法来进行排序。

```csharp
List<MyCustomType> list = new List<MyCustomType>();

// 添加元素到列表
list.Add(new MyCustomType { Id = 2, Name = "B" });
list.Add(new MyCustomType { Id = 1, Name = "A" });

// 排序
list.Sort();
```

这样，你就可以在 `List<T>` 中使用自定义的排序规则了。

总结：

- 使用 `IEquatable<T>` 和 `GetHashCode` 方法来自定义相等性规则和哈希码规则。
- 使用 `IComparable<T>` 接口来自定义排序规则，或者结合 `List<T>.Sort` 方法进行排序。

## 实践（一）——HashSet

1. 创建一个Student类，包含Id，Name两个属性

   ```csharp
   internal class Student
   {
       public int Id { get; set; }
       public string? Name { get; set; }
   
       public Student() { }
       public Student(int id, string name)
       {
           Id = id;
           Name = name;
       }
   }
   ```

2. 新建一个 StudentComparerRule 类，实现 `IEquatable<T>` 接口，并重写 `GetHashCode` 和 `Equals` 方法，自定义相等性规则和哈希码规则

   ```csharp
   // 自定义比较规则 1
   internal class StudentComparerRule : IEqualityComparer<Student>
   {
       // 比较规则的方法
       public bool Equals(Student? x, Student? y)
       {
           // 如果名字一样，则重复
           return x.Name.Equals(y.Name);
       }
   
       // 自定义哈希码
       public int GetHashCode([DisallowNull] Student obj)
       {
           return base.GetHashCode();
       }
   }
   ```

3. 实例化HashSet时候，传入自定义比较规则（实现了IEqualityComparer<T>的类）

   ```csharp
   [Test]
   public void TestSortedClassType()
   {
       HashSet<Student> students = new HashSet<Student>();
       students.Add(new Student(1, "爱莉"));
       students.Add(new Student(2, "爱莉"));
       students.Add(new Student(2, "流萤"));
       students.Add(new Student(2, "流萤"));
       students.Add(new Student(3, "希儿"));
       students.Add(new Student(4, "符玄"));
   
       // 默认情况下，所有的属性值都一样，则表示重复
       foreach (var student in students)
       {
           Console.WriteLine($"Student ID:{student.Id},Name:{student.Name}");
       }
   
       Console.WriteLine("====================");
   
       // 创建一个HashSet，比较名字是否相同
       HashSet<Student> students2 = new HashSet<Student>(new StudentComparerRule()); //传入自定义比较规则
       students2.Add(new Student(1, "爱莉"));
       students2.Add(new Student(2, "爱莉"));
       students2.Add(new Student(2, "流萤"));
       students2.Add(new Student(2, "流萤"));
       students2.Add(new Student(3, "希儿"));
       students2.Add(new Student(4, "符玄"));
   
       foreach (var student in students)
       {
           Console.WriteLine($"Student ID:{student.Id},Name:{student.Name}");
       }
   }
   ```

4. 运行测试，查看控制台输出结果，是否按名字进行了去重，若是则说明我们的自定义比较规则使能。

## 实践（二）——SortedSet

1. 创建一个Student类，包含Id，Name两个属性

   ```csharp
   internal class Student
   {
       public int Id { get; set; }
       public string? Name { get; set; }
   
       public Student() { }
       public Student(int id, string name)
       {
           Id = id;
           Name = name;
       }
   }
   ```

2. 新建一个 StudentCompareRule2 类，实现 `IComparer<T>` 接口，并重写 `Compare` 方法，自定义比较规则

   ```csharp
   // 自定义比较规则 2
   internal class StudentCompareRule2 : IComparer<Student>
   {
       public int Compare(Student? x, Student? y)
       {
           // 返回值>0,表示x大,返回值=0,表示x=y,返回值<0,表示y大
           return x.Id - y.Id;
       }
   }
   ```

3. 实例化SortedSet时候，传入自定义比较规则（实现了IComparer<T>的类）

   ```csharp
   [Test]
   public void TestSortedClassType2()
   {
       SortedSet<Student> students = new SortedSet<Student>(new StudentCompareRule2());
       students.Add(new Student(1, "爱莉"));
       students.Add(new Student(2, "爱莉"));
       students.Add(new Student(2, "流萤"));
       students.Add(new Student(2, "流萤"));
       students.Add(new Student(3, "希儿"));
       students.Add(new Student(4, "符玄"));
   
       foreach (var student in students)
       {
           Console.WriteLine($"Student ID:{student.Id},Name:{student.Name}");
       }
   }
   ```

4. 运行测试，查看控制台输出结果，是否按id从小到大的顺序进行了排序，且去除了id重复的元素，若是则说明我们的自定义比较规则使能。

## 总结

如果需要实现类类型的排序，则需要实现相应的接口（如IComparer<T>、IEqualityComparer<T>），并在实例化集合的时候传入排序的规则（实现了相应接口的类）。
