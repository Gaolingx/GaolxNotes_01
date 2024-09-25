using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GaolxORM
{
    //我们可以把类型当作一种参数（泛型的作用）
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
    }
}
