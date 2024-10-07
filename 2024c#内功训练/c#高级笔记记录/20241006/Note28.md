# C#高级编程之——泛型（八）DataTable转List

在上次手写orm框架的时候，我们在写查询方法时候在DbHelper里用到了一个叫 GetList<T> 的方法，不知道大家还有没有印象，我们今天就来研究下我们如何通过这个方法实现DataTable转List的行为的。

DbHelper中的实现如下：

```csharp
/// <summary>
/// 3. 根据SQL和泛型方法返回泛型【集合】
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="sql"></param>
/// <param name="paras"></param>
/// <returns></returns>
public static List<T> GetList<T>(string sql, params MySqlParameter[]? paras) where T : class, new()
{
    DataTable? dt = null;
    dt = GetDataTable(sql, paras);
    return DataTableExtension.ToList<T>(dt);
}
```

我们可以看到DataTableExtension中有一个ToList<T>的方法，该方法长这样：

1. 设计思路：
   通过分析List<T>与DataTable的对应关系我们发现，由于我们是将数据库中的表格（DataTable）中的字段与对象的属性进行映射，即数据库中的每条数据（记录）都对应List集合中的一个对象，因此我们可以通过遍历DataTable中的所有行获取其中列，再对其中（T）的属性进行赋值并添加到List集合最后返回即可。

```csharp
public static List<T> ToList<T>(DataTable dt) where T : class, new()
{
    Type t = typeof(T); //获取类型
    PropertyInfo[] propertys = t.GetProperties(); //获取所有属性
    List<T> lst = new List<T>();
    string typeName = string.Empty;

    foreach (DataRow dr in dt.Rows) //遍历表格所有行
    {
        T entity = new T(); //创建对象，注：要有new() 这样的一个泛型约束，等价于Activator.CreateInstance<T>()
        foreach (PropertyInfo pi in propertys) //循环对属性赋值
        {
            typeName = pi.Name;
            if (dt.Columns.Contains(typeName))
            {
                if (!pi.CanWrite) continue;
                object value = dr[typeName];
                if (value == DBNull.Value) continue;
                if (pi.PropertyType == typeof(string))
                {
                    pi.SetValue(entity, value.ToString(), null);
                }
                else if (pi.PropertyType == typeof(int) ||
                         pi.PropertyType == typeof(int?))
                {
                    pi.SetValue(entity, int.Parse(value.ToString()), null);
                }
                else if (pi.PropertyType == typeof(DateTime?) ||
                         pi.PropertyType == typeof(DateTime))
                {
                    pi.SetValue(entity, DateTime.Parse(value.ToString()), null);
                }
                else if (pi.PropertyType == typeof(float))
                {
                    pi.SetValue(entity, float.Parse(value.ToString()), null);
                }
                else if (pi.PropertyType == typeof(double))
                {
                    pi.SetValue(entity, double.Parse(value.ToString()), null);
                }
                else
                {
                    pi.SetValue(entity, value, null);
                }
            }
        }

        lst.Add(entity);
    }

    return lst;
}
```
