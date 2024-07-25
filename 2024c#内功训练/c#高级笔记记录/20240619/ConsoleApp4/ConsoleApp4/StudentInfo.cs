
namespace Info.Main
{
    public class StudentInfo
    {

        //无参无返回值的方法
        public void Run()
        {
            Console.WriteLine($"我的名字是{Name}，是米游社的一名创作者");
        }
        //有参无返回值方法
        public void Run2(int age)
        {
            Console.WriteLine($"我的名字是{Name}，我今年{age}岁了");
        }
        //有参有返回值私有方法
        public string Run3(string name)
        {
            return $"我的名字是{name}，我是一个私有方法";
        }

        //构造函数
        public StudentInfo()
        {

        }

        public StudentInfo(string name, int age)
        {
            Name = name;
            Age = age;
        }

        private string _studentId; //字段
        private int _Id { get; set; } //属性（访问器），get set实际上是两个方法
        public int Age { get; set; }
        public string Name { get; set; }

        private int _money;
        private int money
        {
            get { return _money; }
            set { _money = value; }
        }

    }
}