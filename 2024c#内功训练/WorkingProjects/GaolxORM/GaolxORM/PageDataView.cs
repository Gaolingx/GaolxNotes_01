using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GaolxORM
{
    /// <summary>
    /// 分页查询获取数据实体
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class PageDataView<T>
    {
        private int _TotalNum;
        public PageDataView()
        {
            this._Items = new List<T>();
        }
        /// <summary>
        /// 总数
        /// </summary>
        public int TotalNum
        {
            get { return _TotalNum; }
            set { _TotalNum = value; }
        }

        private IList<T> _Items;
        /// <summary>
        /// 具体数据列表
        /// </summary>
        public IList<T> Items
        {
            get { return _Items; }
            set { _Items = value; }
        }
        /// <summary>
        /// 当前页数
        /// </summary>
        public int CurrentPage { get; set; }
        /// <summary>
        /// 总页数
        /// </summary>
        public int TotalPageCount { get; set; }
    }

    /// <summary>
    /// 分页实体
    /// </summary>
    public class PageCriteria
    {
        public PageCriteria()
        {
            ParameterList = new List<ParameterDict>();
        }
        /// <summary>
        /// 查询的表名
        /// </summary>
        public string TableName { get; set; }

        /// <summary>
        /// 字段集合
        /// </summary>
        public string Fields { get; set; }

        /// <summary>
        /// 主键名称
        /// </summary>
        //public string PrimaryKey { get; set; }

        /// <summary>
        /// 每页数量
        /// </summary>
        public int PageSize { get; set; }

        /// <summary>
        /// 当前页码
        /// </summary>
        public int CurrentPage { get; set; }

        /// <summary>
        /// 排序字段
        /// </summary>
        public string Sort { get; set; }

        /// <summary>
        /// 查询条件
        /// </summary>
        public string Condition { get; set; }

        /// <summary>
        /// 总数
        /// </summary>
        public int RecordCount { get; set; }
        /// <summary>
        /// 传入的参数列表
        /// </summary>
        public IList<ParameterDict> ParameterList { get; set; }
    }

    /// <summary>
    /// 参数字典
    /// </summary>
    public class ParameterDict
    {
        /// <summary>
        /// 参数名称
        /// </summary>
        public string ParamName { get; set; }
        /// <summary>
        /// 参数值
        /// </summary>
        public object ParamValue { get; set; }
    }
}
