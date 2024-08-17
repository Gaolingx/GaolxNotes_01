
// 扩展方法的工具类,注意一定要写成static 静态类
namespace ConsoleApp4.Utils
{
    public static class ExtendUtils
    {
        /**
         * 作者：Gaolingx
         * 功能：将Object类型 提供一个转换为Int类型的扩展方法
         * 注意：
         * 1. 静态类下所有的方法都只能是static 方法
         * 2. 把你需要扩展的类型前面加this 
         * 操作步骤：
         * 1. 将该拓展类设置为static 类
         * 2. 创建一个static 方法
         * 3. 在需要扩展的类型前面加一个this 关键字
         */
        public static int ParseInt(this string str) //需要扩展的类型 + 参数的值
        {
            if (string.IsNullOrWhiteSpace(str))
            {
                return 0;
            }

            int result = 0;

            if (!int.TryParse(str, out result)) //TryParse返回值为ture则表示str可以被转换成int类型，返回输出参数out result的值
            {
                return 0;
            }

            return result;
        }
    }
}
