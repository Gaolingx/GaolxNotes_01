
namespace ConsoleApp4
{
    // 元特性：对特性进行描述的特性（特性的特性），可以有多个
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = true)]
    //AttributeTargets用于标识特性的作用范围，AttributeTargets.All表示对所有目标生效。

    public class MyDescriptionAttribute : Attribute
    {
        public string Name { get; set; }
    }
}
