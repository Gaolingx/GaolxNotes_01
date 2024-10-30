# C#高级编程之——泛型（八）ORM框架实现

## 五、ORM框架实现

**5.1 查询**

sql语句：
SELECT 列名1, 列名2, ...  
FROM 表名  
WHERE 条件;

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
    dt = GetDataTable(sql, CommandType.Text, paras);
    return DataTableExtension.ToList<T>(dt);
}

public static List<T> GetList<T>(string sql, CommandType cmdType = CommandType.Text, params MySqlParameter[]? paras) where T : class, new()
{
    DataTable? dt = null;
    dt = GetDataTable(sql, cmdType, paras);
    return DataTableExtension.ToList<T>(dt);
}
```

**5.2 添加**

sql语句：
INSERT INTO 表名 (列1, 列2, 列3, ...)  
VALUES (值1, 值2, 值3, ...);

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

**5.3 更新**

sql语句：
UPDATE 表名  
SET 列名1 = 值1, 列名2 = 值2, ...  
WHERE 条件;

```csharp

/// <summary>
/// 更新功能
/// </summary>
/// <param name="model"></param>
public int Update(T model)
{
    var tp = typeof(T);
    var pk = GetPrimaryKey(); //获取主键
    var props = tp.GetProperties().ToList();
    //获取所有的属性名称(除主键)
    var propNames = props.Where(p => !p.Name.Equals(pk)).Select(p => p.Name).ToList();


    //update 表 set 字段1=@字段1,字段2=@字段2, where 主键名=主键值
    string sql = $"update {tp.Name} set ";
    foreach (var propName in propNames)
    {
        sql += $"{propName}=@{propName},";
    }

    sql = sql.Remove(sql.Length - 1);

    sql += $" where {pk.Name}=@{pk.Name}";

    List<SqlParameter> list = new();
    foreach (var prop in props)
    {
        SqlParameter parameter = new SqlParameter(prop.Name, prop.GetValue(model));
        list.Add(parameter);
    }

    return DbHelper.ExecuteNonQuery(sql, list.ToArray());
}

/// <summary>
/// 获取主键
/// </summary>
/// <returns></returns>
public PropertyInfo GetPrimaryKey()
{
    var props = typeof(T).GetProperties();
    foreach (var propertyInfo in props)
    {
        //获取特性
        var attrs = propertyInfo.GetCustomAttributes(typeof(KeyAttribute), false);
        if (attrs.Length > 0)
        {
            return propertyInfo;
        }
    }

    return props[0]; // 如果没有Key 特性，就让第一个属性当作主键
}
```

**5.4 删除**

sql语句：
DELETE FROM students  
WHERE name = '张三';

```csharp
/// <summary>
/// 删除功能
/// </summary>
/// <param name="id"></param>
public int Delete(dynamic id)
{
    //delete from 表名 where 主键名=@主键值

    var pk = GetPrimaryKey().Name;
    return DbHelper.ExecuteNonQuery($"delete from {typeof(T).Name} where {pk}=@{pk}", new SqlParameter(pk, id));
}
```

**5.5 查询拓展（根据主键）**

sql语句：
SELECT 列名1, 列名2, ...  
FROM 表名  
WHERE 条件;

```csharp
/// <summary>
/// 查询功能
/// </summary>
/// <param name="id"></param>
/// <returns></returns>
public T GetModel(dynamic id) //id为dynamic类型，因为主键的类型通常是不确定的（例如可能是int,也有可能是string,long）
{
    var pk = GetPrimaryKey().Name; //获取主键的名称
                                   //获取一条记录
    return DbHelper.GetList<T>(
        $"select * from {typeof(T).Name} where {pk}=@id",
        new SqlParameter(pk, id)).First();
}
```
