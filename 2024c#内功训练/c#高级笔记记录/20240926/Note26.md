# C#高级编程之——泛型（七）ORM框架搭建（下）

## 四、ORM框架地基搭建

### 4.1 orm如何实现对数据库的增删查改功能

1. 获取类型反射生成sql语句
2. ado.net 执行生成的sql语句

如此便简化了增删查改的流程，只需调用增删查改对应的方法即可实现调用数据库相关操作，同时避免书写sql语句出错的可能，也不用重复书写sql语句。

### 4.2 具体如何操作（实现思路）

1. 我们可以将类型当作一种参数传递到我们DbContext的泛型方法中（例如 Add()）
2. 在DbContext对应增删查改的泛型方法中获取传入参数的类型
3. 根据类型得到属性，拼接sql语句
4. 调用数据库相关方法操作数据表

以上操作就被称之为：对象——关系映射（ORM框架），即将c#中的实体对象（对象中的属性）与数据表中的字段进行一一对应。

注：为了方便演示，本教程假定表名和类名一致，字段名与属性名一致，你也可以通过特性实现更加复杂的映射。

### 4.3 常见的ORM框架

Dapper、EntityFramework Core

### 4.4 数据库操作部分

为了让简化数据库的操作，我们先新建一个DbHelper类帮助我们实现对数据库的操作：

```csharp
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using System.Collections;
using MySql.Data.MySqlClient;

namespace GaolxORM
{
    /// <summary>
    /// 数据库帮助类
    /// </summary>
    public class DbHelper
    {
        private static string? ConnectionString { get; }

        static DbHelper()
        {
            ConnectionString = ConfigurationManager.ConnectionStrings["connString"].ConnectionString;
        }

        /// <summary>
        /// 1. 执行添加、删除、修改通用方法
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static int ExecuteNonQuery(string sql, params MySqlParameter[]? paras)
        {

            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                //打开数据库连接
                conn.Open();
                //创建执行脚本的对象
                MySqlCommand command = new MySqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                int result = command.ExecuteNonQuery();
                return result;
            }
        }

        public static MySqlCommand ExecuteNonQueryCmd(string sql, params MySqlParameter[]? paras)
        {

            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                //打开数据库连接
                conn.Open();
                //创建执行脚本的对象
                MySqlCommand command = new MySqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                command.ExecuteNonQuery();
                return command;
            }
        }

        /// <summary>
        /// 2. 执行SQL并返回第一行第一列
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static object ExecuteScalar(string sql, params MySqlParameter[]? paras)
        {
            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                conn.Open();
                MySqlCommand command = new MySqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                object obj = command.ExecuteScalar();
                return obj;
            }
        }

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

        /// <summary>
        /// 4. 查询返回临时表
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static DataTable GetDataTable(string sql, params MySqlParameter[]? paras)
        {
            DataTable? dt = null;
            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                MySqlCommand command = new MySqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                MySqlDataAdapter adapter = new MySqlDataAdapter(command);
                dt = new DataTable();
                adapter.Fill(dt);
            }
            return dt;
        }

        /// <summary>
        /// 5. 执行SQL返回MySqlDataReader对象（游标）
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static MySqlDataReader ExecuteReader(string sql, params MySqlParameter[]? paras)
        {
            MySqlConnection conn = new MySqlConnection(ConnectionString);
            conn.Open();
            MySqlCommand command = new MySqlCommand(sql, conn);
            command.Parameters.AddRange(paras);
            return command.ExecuteReader(CommandBehavior.CloseConnection);//命令行为
        }

        /// <summary>
        /// 6. 根据SQL执行返回数据集(多临时表)
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static DataSet GetDataSet(string sql, params MySqlParameter[]? paras)
        {
            DataSet? ds = null;
            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                MySqlCommand command = new MySqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                MySqlDataAdapter adapter = new MySqlDataAdapter(command);
                ds = new DataSet();
                adapter.Fill(ds);
            }
            return ds;
        }

        /// <summary>
        /// 7. 执行事务的通用方法
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static bool ExecuteTransaction(string[] sql, params MySqlParameter[]? paras)
        {
            bool result = false;
            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                conn.Open();
                MySqlCommand command = new MySqlCommand();
                command.Parameters.AddRange(paras);
                command.Connection = conn;//关联联接对象
                command.Transaction = conn.BeginTransaction();//开始事务
                try
                {
                    for (int i = 0; i < sql.Length; i++)
                    {
                        command.CommandText = sql[i];
                        command.ExecuteNonQuery();//执行
                    }
                    command.Transaction.Commit();//提交
                    result = true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error:{ex}");
                    command.Transaction.Rollback();//回滚
                    result = false;
                }
            }
            return result;
        }

        /// <summary>
        /// 8. 事务批量添加
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="list"></param>
        /// <returns></returns>
        public static bool ExecuteTransaction(string[] sql, List<MySqlParameter[]> list)
        {
            bool result = false;
            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                conn.Open();
                MySqlCommand command = new MySqlCommand();
                foreach (MySqlParameter[] item in list)
                {
                    command.Parameters.AddRange(item);
                }
                command.Connection = conn;//关联联接对象
                command.Transaction = conn.BeginTransaction();//开始事务
                try
                {
                    for (int i = 0; i < sql.Length; i++)
                    {
                        command.CommandText = sql[i];
                        command.ExecuteNonQuery();//执行
                    }
                    command.Transaction.Commit();//提交
                    result = true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error:{ex}");
                    command.Transaction.Rollback();//回滚
                    result = false;
                }
            }
            return result;
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
