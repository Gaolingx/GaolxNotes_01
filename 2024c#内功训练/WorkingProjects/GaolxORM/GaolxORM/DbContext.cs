using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
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
        public List<T> Select(Expression<Func<T,bool>> expression = null) //TODO:表达式树，用于动态生成查询条件
        {
            return DbHelper.GetList<T>($"select * from {typeof(T).Name}"); //这里假定表名与类名映射
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
