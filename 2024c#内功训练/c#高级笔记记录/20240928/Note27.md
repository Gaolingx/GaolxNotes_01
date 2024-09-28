# C#高级编程之——泛型（七）ORM框架实现

## 五、ORM框架实现

**5.1 查询**

sql语句：select * from 表名 (where 条件)

```csharp
/// <summary>
/// 查询功能
/// </summary>
/// <returns></returns>
public List<T> Select(Expression<Func<T,bool>> expression = null) //TODO:表达式树，用于动态生成查询条件
{
    return DbHelper.GetList<T>($"select * from {typeof(T).Name}"); //这里假定表名与类名映射
}
```

DbHelper中的实现如下：

```csharp
public static List<T> GetList<T>(string sql, params SqlParameter[] paras)
{
    List<T> list = new List<T>();
    using (SqlConnection conn = new SqlConnection(ConnectionString))
    {
        conn.Open();
        SqlCommand command = new SqlCommand(sql, conn);
        command.Parameters.AddRange(paras);
        using (SqlDataReader reader = command.ExecuteReader())
        {   //typeof()检测类型
            Type type = typeof(T);//类型的声明(可声明一个不确定的类型)
            while (reader.Read())
            {
                T t = (T)Activator.CreateInstance(type);
                //通过反射去遍历属性
                foreach (PropertyInfo info in type.GetProperties())
                {
                    info.SetValue(t, reader[info.Name] is DBNull ?
                                            null : reader[info.Name]);
                }
                list.Add(t);
            }
        }
    }
    return list;//命令行为
}
```

**5.2 添加**

sql语句：insert into Student(...) values(...)

```csharp
/// <summary>
/// 添加功能
/// </summary>
/// <param name="model">要添加的对象</param>
public void Add(T model, int skipCount = 1) //这里的model是要添加的实体对象
{
    // insert into Student(.属性1,属性2,属性3..) values(.'属性值1','属性值2'，'属性值3'..)
    StringBuilder sql = new StringBuilder($"insert into {typeof(T).Name}(");
    // 跳过第一个属性，忽略主键（你也可以通过attribute标注key）
    var propertyInfos = typeof(T).GetProperties().Skip(skipCount);
    // 获取所有属性名
    var propNames = propertyInfos.Select(p => p.Name).ToList(); //我们只需要propertyInfo的name
    //将属性名拼接到sql语句中
    sql.Append(string.Join(",", propNames));
    sql.Append(") values('");

    List<string> values = new List<string>();
    foreach (var propertyInfo in propertyInfos)
    {
        // 获取属性值
        values.Add(propertyInfo.GetValue(model).ToString());
    }
    sql.Append(string.Join("','", values));
    sql.Append("')");
    DbHelper.ExecuteNonQuery(sql.ToString());
}
```
