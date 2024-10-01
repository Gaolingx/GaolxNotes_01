using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using System.Collections;
using System.Reflection;

namespace GaolxORM
{
    /// <summary>
    /// 数据库帮助类
    /// </summary>
    public abstract class DbHelper
    {
        public DbHelper()
        {
            ConnectionString = ConfigurationManager.ConnectionStrings["connString"].ConnectionString;
        }

        public static string? ConnectionString { get; set; }

        /// <summary>
        /// 1. 执行添加、删除、修改通用方法
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static int ExecuteNonQuery(string sql, params SqlParameter[]? paras)
        {

            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                //打开数据库连接
                conn.Open();
                //创建执行脚本的对象
                SqlCommand command = new SqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                int result = command.ExecuteNonQuery();
                return result;
            }
        }

        /// <summary>
        /// 2. 执行SQL并返回第一行第一列
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static object ExecuteScalar(string sql, params SqlParameter[]? paras)
        {
            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                conn.Open();
                SqlCommand command = new SqlCommand(sql, conn);
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
        public static List<T> GetListByColumns<T>(string sql, params SqlParameter[]? paras) where T : class, new()
        {
            List<T> list = new List<T>();
            //获取select 和form中间的字段
            string newSql = sql.ToLower();
            string columnStr = newSql.Substring(newSql.IndexOf("select ") + 7, newSql.IndexOf(" from") - 7)
                .Replace(" ", "")
                .Replace("\r\n", "")
                .Replace("[", "")
                .Replace("]", "");
            //保存字段
            ArrayList columns = new ArrayList(columnStr.Split(','));
            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                conn.Open();
                SqlCommand command = new SqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                using (SqlDataReader reader = command.ExecuteReader())
                {
                    //typeof()检测类型
                    Type type = typeof(T);//类型的声明(可声明一个不确定的类型)

                    if (columnStr == "*")//查询所有(里面不用判断)
                    {
                        while (reader.Read())
                        {
                            T? t = Activator.CreateInstance(type) as T;
                            //通过反射去遍历属性
                            foreach (PropertyInfo info in type.GetProperties())
                            {
                                info.SetValue(t, reader[info.Name] is DBNull ?
                                                        null : reader[info.Name]);
                            }
                            list.Add(t);
                        }
                    }
                    else//根据查询的列遍历
                    {
                        while (reader.Read())
                        {
                            T? t = Activator.CreateInstance(type) as T;
                            //通过反射去遍历属性
                            foreach (PropertyInfo info in type.GetProperties())
                            {
                                if (columns.Contains(info.Name.ToLower()))//判断是否存在
                                {
                                    info.SetValue(t, reader[info.Name] is DBNull ?
                                                            null : reader[info.Name]);
                                }
                            }
                            list.Add(t);
                        }
                    }
                }
            }
            return list;//命令行为
        }

        public static List<T> GetList<T>(string sql, params SqlParameter[]? paras) where T : class, new()
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
        public static DataTable GetDataTable(string sql, params SqlParameter[]? paras)
        {
            DataTable? dt = null;
            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                SqlCommand command = new SqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                SqlDataAdapter adapter = new SqlDataAdapter(command);
                dt = new DataTable();
                adapter.Fill(dt);
            }
            return dt;
        }

        /// <summary>
        /// 5. 执行SQL返回SqlDataReader对象（游标）
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static SqlDataReader ExecuteReader(string sql, params SqlParameter[]? paras)
        {
            SqlConnection conn = new SqlConnection(ConnectionString);
            conn.Open();
            SqlCommand command = new SqlCommand(sql, conn);
            command.Parameters.AddRange(paras);
            return command.ExecuteReader(CommandBehavior.CloseConnection);//命令行为
        }

        /// <summary>
        /// 6. 根据SQL执行返回数据集(多临时表)
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static DataSet GetDataSet(string sql, params SqlParameter[]? paras)
        {
            DataSet? ds = null;
            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                SqlCommand command = new SqlCommand(sql, conn);
                command.Parameters.AddRange(paras);
                SqlDataAdapter adapter = new SqlDataAdapter(command);
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
        public static bool ExecuteTransaction(string[] sql, params SqlParameter[]? paras)
        {
            bool result = false;
            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                conn.Open();
                SqlCommand command = new SqlCommand();
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
        public static bool ExecuteTransaction(string[] sql, List<SqlParameter[]> list)
        {
            bool result = false;
            using (SqlConnection conn = new SqlConnection(ConnectionString))
            {
                conn.Open();
                SqlCommand command = new SqlCommand();
                foreach (SqlParameter[] item in list)
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
