# C#高级编程之——泛型（七）ORM框架搭建（下）

## 四、ORM框架实现

4.1 orm如何实现对数据库的增删查改功能：

1. 获取类型反射生成sql语句
2. ado.net 执行生成的sql语句

如此便简化了增删查改的流程，只需调用增删查改对应的方法即可实现调用数据库相关操作，同时避免书写sql语句出错的可能，也不用重复书写sql语句。

4.2 具体如何操作（实现思路）：

1. 我们可以将类型当作一种参数传递到我们DbContext的泛型方法中（例如 Add()）
2. 在DbContext对应增删查改的泛型方法中获取传入参数的类型
3. 根据类型得到属性，拼接sql语句
4. 调用数据库相关方法操作数据表

以上操作就被称之为：对象——关系映射（ORM框架），即将c#中的实体对象（对象中的属性）与数据表中的字段进行一一对应。

注：为了方便演示，本教程假定表名和类名一致，字段名与属性名一致，你也可以通过特性实现更加复杂的映射。

4.3 常见的ORM框架：Dapper、EntityFramework Core

4.4 数据库操作部分

为了让简化数据库的操作，我们先新建一个DbHelper类帮助我们实现对数据库的操作：

```csharp
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.Common;
using System.Configuration;

namespace GaolxORM
{
    //帮助类的基类(抽象类)
    public abstract class DbHelper
    {
        public DbHelper()
        {
            ConnectionString = ConfigurationManager.ConnectionStrings["connString"].ConnectionString;
        }

        public abstract DbProviderFactory DbProviderFactory { get; }

        public string ConnectionString { get; }

        public List<T> GetList<T>(string sql, params SqlParameter[] parameters)
        {
            using (var connection = new SqlConnection(ConnectionString))
            {
                connection.Open();
                using (var command = new SqlCommand(sql, connection))
                {
                    if (parameters != null)
                    {
                        command.Parameters.AddRange(parameters);
                    }

                    using (var reader = command.ExecuteReader())
                    {
                        return reader.Cast<IDataRecord>()
                            .Select(r => ConvertToEntity<T>(r))
                            .ToList();
                    }
                }
            }
        }

        private T ConvertToEntity<T>(IDataRecord record)
        {
            var entity = Activator.CreateInstance<T>();
            var properties = typeof(T).GetProperties();
            foreach (var prop in properties)
            {
                if (record.FieldCount > 0 && record.GetName(record.GetOrdinal(prop.Name)) != null)
                {
                    prop.SetValue(entity, record[prop.Name]);
                }
            }
            return entity;
        }

        public int ExecuteNonQuery(string sql, params SqlParameter[] parameters)
        {
            using (var connection = new SqlConnection(ConnectionString))
            using (var command = new SqlCommand(sql, connection))
            {
                command.Parameters.AddRange(parameters);
                connection.Open();
                return command.ExecuteNonQuery();
            }
        }

        public object ExecuteScalar(string sql, params SqlParameter[] parameters)
        {
            using (var connection = new SqlConnection(ConnectionString))
            using (var command = new SqlCommand(sql, connection))
            {
                command.Parameters.AddRange(parameters);
                connection.Open();
                return command.ExecuteScalar();
            }
        }

        public SqlDataReader ExecuteReader(string sql, params SqlParameter[] parameters)
        {
            var connection = new SqlConnection(ConnectionString);
            var command = new SqlCommand(sql, connection);
            command.Parameters.AddRange(parameters);
            connection.Open();
            return command.ExecuteReader(CommandBehavior.CloseConnection);
        }

        public DataTable FillDataTable(string sql, params SqlParameter[] parameters)
        {
            using (var connection = new SqlConnection(ConnectionString))
            using (var command = new SqlCommand(sql, connection))
            using (var adapter = new SqlDataAdapter(command))
            {
                command.Parameters.AddRange(parameters);
                var dataTable = new DataTable();
                adapter.Fill(dataTable);
                return dataTable;
            }
        }
    }
}

```

4.5 配置文件部分

我们需要将连接数据库的connectionString写到config中，项目中新建一个App.config的配置文件，写入如下信息

```xml
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <connectionStrings>
    <add name="connString" connectionString="server=localhost;User Id = root;password=123456;Database=test;Charset = utf8mb3"></add>
  </connectionStrings>
</configuration>
```

最后整个框架雏形如下，至此，我们完成了orm基本框架的搭建

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GaolxORM
{
    public class DbContext<T> where T : class, new()
    {
        /// <summary>
        /// 添加功能
        /// </summary>
        /// <param name="model">要添加的对象</param>
        public void Add(T model) //这里的model是要添加的实体对象
        {

        }

        /// <summary>
        /// 修改功能
        /// </summary>
        /// <param name="model"></param>
        public void Update(T model)
        {

        }

        /// <summary>
        /// 查询功能
        /// </summary>
        /// <returns></returns>
        public List<T> GetList()
        {
            return null;
        }

        /// <summary>
        /// 编辑功能（根据主键得到实体）
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        public T GetModel(dynamic id) //id为dynamic类型，因为主键的类型通常是不确定的（例如可能是int,也有可能是string,long）
        {
            return null;
        }

        /// <summary>
        /// 删除功能
        /// </summary>
        /// <param name="id"></param>
        public void Delete(dynamic id)
        {

        }
    }
}

```
