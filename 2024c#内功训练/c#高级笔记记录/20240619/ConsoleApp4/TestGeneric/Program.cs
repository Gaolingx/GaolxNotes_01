
namespace TestGeneric
{
    public class MyGenericClass01<T> //<T> 代指泛型，泛指某种类型，类型在编译期间确定，T也可以用其他字符标识
    {
        public T ItemName { get; set; }
        public int Total {  get; set; }
    }

    public class TestClass
    {
        MyGenericClass01<int> num = new MyGenericClass01<int>(); //T 代表int类型

        MyGenericClass01<string> str = new MyGenericClass01<string>(); //T 代表string类型
        private void Test()
        {
            num.ItemName = 10; //这里的ItemName变成了int类型
            str.ItemName = "爱莉小跟班gaolx"; //这里的ItemName变成了string类型
        }
    }
}
