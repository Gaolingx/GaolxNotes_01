using MySql.Data.MySqlClient;
using System.Configuration;
using System.Data;

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
            var config = ConfigurationManager.OpenExeConfiguration(ConfigurationUserLevel.None);
            var appSettings = config.AppSettings;
            string connStr = appSettings.Settings["connString"].Value;
            string dataBase = appSettings.Settings["dataBase"].Value;

            ConnectionString = $"{connStr}{dataBase}";
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
            dt = GetDataTable(sql, CommandType.Text, paras);
            return DataTableExtension.ToList<T>(dt);
        }

        public static List<T> GetList<T>(string sql, CommandType cmdType = CommandType.Text, params MySqlParameter[]? paras) where T : class, new()
        {
            DataTable? dt = null;
            dt = GetDataTable(sql, cmdType, paras);
            return DataTableExtension.ToList<T>(dt);
        }

        /// <summary>
        /// 4. 查询返回临时表
        /// </summary>
        /// <param name="sql"></param>
        /// <param name="paras"></param>
        /// <returns></returns>
        public static DataTable GetDataTable(string sql, CommandType cmdType = CommandType.Text, params MySqlParameter[]? paras)
        {
            DataTable? dt = null;
            using (MySqlConnection conn = new MySqlConnection(ConnectionString))
            {
                MySqlCommand command = new MySqlCommand(sql, conn);
                command.CommandType = cmdType;
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
