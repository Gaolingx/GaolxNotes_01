using System.ComponentModel;
using System.Reflection;


namespace TestGetDescription
{
    public static class EnumExtend
    {
        /// <summary>
        /// 获取枚举的字段的描述
        /// </summary>
        /// <param name="stateEnum"></param>
        /// <returns></returns>
        public static string GetDescription(this OrderStateEnum stateEnum)
        {
            //通过反射获取
            var stateType = typeof(OrderStateEnum);
            //获取当前被操作的枚举字段
            var field = stateType.GetField(stateEnum.ToString(), BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public); //根据字段名称获取字段对象
            var descriptionAttribute = field?.GetCustomAttribute<DescriptionAttribute>();
            //var descriptionAttribute = field?.GetCustomAttribute(typeof(DescriptionAttribute)) as DescriptionAttribute;
            return descriptionAttribute?.Description ?? string.Empty;
        }
    }
}
