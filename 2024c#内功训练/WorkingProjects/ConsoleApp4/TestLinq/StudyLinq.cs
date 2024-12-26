using NUnit.Framework;

namespace TestLinq
{
    internal class StudyLinq
    {
        #region 测试数据
        public static List<StuInfo> GetStudentInfos()
        {
            List<StuInfo> stuInfos = new List<StuInfo>()
            {
                new StuInfo { Id = 1001, Name = "流萤", Sex = "女", Age = 20, Chinese = 100, Math = 120, English = 95, Physics = 70, Score = 500, Grade = "A",GroupId = 1 },
                new StuInfo { Id = 1002, Name = "符玄", Sex = "女", Age = 20, Chinese = 105, Math = 130, English = 100, Physics = 80, Score = 500, Grade = "A",GroupId = 1 },
                new StuInfo { Id = 1003, Name = "爱莉希雅", Sex = "女", Age = 18, Chinese = 110, Math = 90, English = 105, Physics = 65, Score = 500, Grade = "B" ,GroupId = 2},
                new StuInfo { Id = 1003, Name = "琪亚娜", Sex = "女", Age = 19, Chinese = 90, Math = 85, English = 100, Physics = 60, Score = 500, Grade = "B" ,GroupId = 3}
            };
            return stuInfos;
        }

        public static List<ClassGroup> GetClassGroups()
        {
            List<ClassGroup> classGroups = new List<ClassGroup>()
            {
                new ClassGroup{Id = 1,GroupName="组1"},
                new ClassGroup{Id = 2,GroupName="组2"},
                new ClassGroup{Id = 3,GroupName="组3"},
                new ClassGroup{Id = 4,GroupName="组4"},
            };
            return classGroups;
        }

        public static List<StuInfo> GetStudentInfos2()
        {
            List<StuInfo> stuInfos = new List<StuInfo>()
            {
                new StuInfo { Id = 1001, Name = "流萤", Sex = "女", Age = 20, Chinese = 100, Math = 120, English = 95, Physics = 70, Score = 500, Grade = "A",GroupId = 1 },
                new StuInfo { Id = 1002, Name = "符玄", Sex = "女", Age = 20, Chinese = 105, Math = 130, English = 100, Physics = 80, Score = 500, Grade = "A",GroupId = 1 },
                new StuInfo { Id = 1003, Name = "爱莉希雅", Sex = "女", Age = 18, Chinese = 110, Math = 90, English = 105, Physics = 65, Score = 500, Grade = "B" ,GroupId = 2},
                new StuInfo { Id = 1003, Name = "琪亚娜", Sex = "女", Age = 19, Chinese = 90, Math = 85, English = 100, Physics = 60, Score = 500, Grade = "B" ,GroupId = 3},
                new StuInfo { Id = 1003, Name = "派蒙", Sex = "女", Age = 9, Chinese = 80, Math = 95, English = 110, Physics = 60, Score = 500, Grade = "A" ,GroupId = 6} // 分类ID不存在
            };
            return stuInfos;
        }

        public static List<ClassGroup> GetClassGroups2()
        {
            List<ClassGroup> classGroups = new List<ClassGroup>()
            {
                new ClassGroup{Id = 1,GroupName="组1"},
                new ClassGroup{Id = 2,GroupName="组2"},
                new ClassGroup{Id = 3,GroupName="组3"},
                new ClassGroup{Id = 4,GroupName="组4"}, // 不存在任何数据
                new ClassGroup{Id = 5,GroupName="组5"}, // 不存在任何数据
            };
            return classGroups;
        }

        #endregion

        /// <summary>
        /// 查询所有学生信息
        /// </summary>
        [Test]
        public void TestLinq01()
        {
            var stuInfoLst = GetStudentInfos(); // 数据源

            // 语法：var 查询结果 = from 字段 in 数据源 select 字段
            var stuInfos = from prop in stuInfoLst select prop;

            foreach (var stuInfo in stuInfos)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                    $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
            }
        }

        /// <summary>
        /// 查询等级为A的学生信息(Grade = "A")
        /// </summary>
        [Test]
        public void TestLinq02()
        {
            var stuInfoLst = GetStudentInfos(); // 数据源

            // 语法：var 查询结果 = from 字段 in 数据源 where 条件 select 字段
            var stuInfos = from prop in stuInfoLst
                           where prop.Grade == "A" // where接 bool表达式
                           select prop;

            foreach (var stuInfo in stuInfos)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                    $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
            }
        }

        /// <summary>
        /// 查询所有学生信息，并按照id降序排列
        /// </summary>
        [Test]
        public void TestLinq03()
        {
            var stuInfoLst = GetStudentInfos();

            // 语法：var 查询结果 = from 字段 in 数据源 select 字段
            var stuInfos = from prop in stuInfoLst
                           orderby prop.Id descending //descending 降序 ascending升序（默认）
                           select prop;

            foreach (var stuInfo in stuInfos)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                    $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
            }
        }

        /// <summary>
        /// 查询所有学生信息，按照 Grade 进行分组
        /// </summary>
        [Test]
        public void TestLinq04()
        {
            var stuInfoLst = GetStudentInfos();

            // 语法：var 查询结果 = from 字段 in 数据源 group 字段 by 分组条件
            var stuInfos = from prop in stuInfoLst
                           group prop by prop.Grade;

            foreach (var groupItem in stuInfos)
            {
                Console.WriteLine(groupItem.Key); //分组依据

                foreach (var stuInfo in groupItem) //遍历groupItem，输出其中的每一个元素
                {
                    Console.WriteLine($"   Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                        $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
                }
            }
        }

        /// <summary>
        /// 查询所有学生信息，按照 Age 进行分组，并对每组元素的数量进行升序排列
        /// </summary>
        [Test]
        public void TestLinq05()
        {
            var stuInfoLst = GetStudentInfos();

            // 语法：var 查询结果 = from 字段 in 数据源 group 字段 by 分组条件
            var stuInfos = from prop in stuInfoLst
                           group prop by prop.Age
                           into groupdata //group by分组后，如果要对分组后的每组元素进行操作，需要into关键字重新赋值，g指每组的元素
                           orderby groupdata.Count() ascending
                           select groupdata;

            foreach (var groupItem in stuInfos)
            {
                Console.WriteLine(groupItem.Key); //分组依据

                foreach (var stuInfo in groupItem) //遍历groupItem，输出其中的每一个元素
                {
                    Console.WriteLine($"   Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                        $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
                }
            }
        }

        /// <summary>
        /// 只查询学生的Id，名字和年龄
        /// </summary>
        [Test]
        public void TestLinq06()
        {
            var stuInfoLst = GetStudentInfos();

            var stuLst = from prop in stuInfoLst
                         select new { prop.Id, prop.Name, prop.Age, Remark = "备注" }; //匿名类

            foreach (var stuInfo in stuLst)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Age = {stuInfo.Age}, Remark = {stuInfo.Remark}");
            }
        }

        /// <summary>
        /// 查询年龄大于等于20岁的学生，显示学生Id，名字，总分（Chinese+Math+English+Physics）
        /// </summary>
        [Test]
        public void TestLinq07()
        {
            // 方法一：
            var stuInfoLst = from item in GetStudentInfos()
                             where item.Age >= 20
                             select new { item.Id, item.Name, Total = item.Chinese + item.Math + item.English + item.Physics }; // 需要定义属性接受表达式的指

            foreach (var stuInfo in stuInfoLst)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name},Total Score = {stuInfo.Total}");
            }
            Console.WriteLine("====================");

            // 方法二：
            var stuInfoLst2 = from item in GetStudentInfos()
                              let total = item.Chinese + item.Math + item.English + item.Physics
                              where item.Age >= 20
                              select new { item.Id, item.Name, Total = total }; // 需要定义属性接受表达式的指

            foreach (var stuInfo in stuInfoLst2)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name},Total Score = {stuInfo.Total}");
            }
            Console.WriteLine("====================");
        }

        /// <summary>
        /// 查询学生详情，显示学生名字和所在的组名（分类名称）
        /// </summary>
        [Test]
        public void TestLinq08()
        {
            var stuInfoLst = from p in GetStudentInfos()
                             from g in GetClassGroups()
                             where p.GroupId == g.Id
                             select new { p.Name, g.GroupName };

            foreach (var item in stuInfoLst)
            {
                Console.WriteLine($"Name = {item.Name}, GroupName = {item.GroupName}");
            }
        }

        /// <summary>
        /// 查询学生详情，显示学生名字和所在的组名（通过join 内连接关联）
        /// </summary>
        [Test]
        public void TestLinq09()
        {
            var stuInfoLst = from p in GetStudentInfos2()
                             join g in GetClassGroups2()
                             on p.GroupId equals g.Id //关联数据源，不能用 ==，而是 equals关键字
                             select new { p.Name, g.GroupName };

            foreach (var item in stuInfoLst)
            {
                Console.WriteLine($"Name = {item.Name}, GroupName = {item.GroupName}");
            }
        }

        /// <summary>
        /// 从小组信息中查询小组id，小组名称，以及这个小组下所有学生的数量
        /// </summary>
        [Test]
        public void TestLinq10()
        {
            var groupItems = from g in GetClassGroups2()
                             join p in GetStudentInfos2()
                             on g.Id equals p.GroupId
                             into ps //将分类对象 g 下所有的 p 保存到 ps 中，即ps存储了分类g下所有的学生信息
                             select new { g.Id, g.GroupName, Count = ps.Count() };

            foreach (var item in groupItems)
            {
                Console.WriteLine($"Id = {item.Id}, GroupName = {item.GroupName}, Student Count:{item.Count}");
            }
        }

        /// <summary>
        /// 既要查询学生信息，要求显示学生的id，学生名字，小组名字。
        /// </summary>
        [Test]
        public void TestLinq11()
        {
            // 情况一：有学生未分组（左表：Students）
            var stuInfos = from students in GetStudentInfos2()
                           join classGroup in GetClassGroups2()
                           on students.GroupId equals classGroup.Id
                           into cs
                           // 从分类组 cs 中获取分类信息，如果（右数据集合）没有匹配的元素则使用 默认值：class null，或者new一个分类对象，指定它的初始值
                           from c2 in cs.DefaultIfEmpty(new ClassGroup() { GroupName = "无" }) // 有分类：显示分类，无分类：显示无
                           select new { students.Id, students.Name, c2.GroupName };

            foreach (var item in stuInfos)
            {
                Console.WriteLine($"Id = {item.Id}, Student Name = {item.Name}, Group Name:{item.GroupName}");
            }
            Console.WriteLine("====================");

            // 情况二：有分组无学生（左表：ClassGroups）

            var groupItems = from classGroup in GetClassGroups2()
                             join students in GetStudentInfos2()
                             on classGroup.Id equals students.GroupId
                             into ps
                             from p2 in ps.DefaultIfEmpty(new StuInfo()) // 当分类没有对应的学生信息时需要做相应的null值处理
                             select new { p2.Id, p2.Name, classGroup.GroupName };

            foreach (var item in groupItems)
            {
                Console.WriteLine($"Id = {item.Id}, Student Name = {item.Name}, Group Name:{item.GroupName}");
            }
        }

        /// <summary>
        /// 查询年龄大于19岁且位于第一小组的学生信息
        /// </summary>
        [Test]
        public void TestLinq12()
        {
            var stuInfos = from p in GetStudentInfos2()
                           where p.Age > 19
                           where p.GroupId == 1
                           // 等价于 where p.Age > 19 && p.GroupId == 1
                           select p;

            foreach (var stuInfo in stuInfos)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                    $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}, Group = {stuInfo.GroupId}");
            }
        }

        /// <summary>
        /// 根据小组id，查询所在组的学生数量
        /// </summary>
        /// <param name="groupId"></param>
        /// <returns></returns>
        private static int GetStudentsCountByGroupId(int groupId)
        {
            int count = GetStudentInfos2().Where(item => item.GroupId == groupId).Count();
            return count;
        }

        /// <summary>
        /// 查询小组的详细信息，显示小组ID，小组名称，小组人数
        /// </summary>
        [Test]
        public void TestLinq13()
        {
            var groupItems = from classGroup in GetClassGroups2()
                             select new { classGroup.Id, classGroup.GroupName, Count = GetStudentsCountByGroupId(classGroup.Id) };

            foreach (var item in groupItems)
            {
                Console.WriteLine($"Group Id = {item.Id}, Group Name = {item.GroupName}, Student Count:{item.Count}");
            }
        }

        /// <summary>
        /// 查询小组的详细信息，显示小组ID，小组名称，小组人数，且只返回小组中的学生个数大于0的查询结果
        /// </summary>
        [Test]
        public void TestLinq14()
        {
            var groupItems = from classGroup in GetClassGroups2()
                             let count = GetStudentsCountByGroupId(classGroup.Id)
                             where count > 0
                             select new { classGroup.Id, classGroup.GroupName, Count = count };

            foreach (var item in groupItems)
            {
                Console.WriteLine($"Group Id = {item.Id}, Group Name = {item.GroupName}, Student Count:{item.Count}");
            }
        }

        private static bool GetStudentAge(StuInfo stuInfo)
        {
            if (stuInfo.Age > 18)
            {
                return true;
            }
            return false;
        }

        /// <summary>
        /// 查询年龄大于18岁的所有学生信息
        /// </summary>
        [Test]
        public void TestLinq15()
        {
            var students = GetStudentInfos2();
            var data = students.Where(delegate (StuInfo val) { return GetStudentAge(val); }); // Func<TSource, bool>
            //var data2 = students.Where(item => item.Age > 19);

            foreach (var stuInfo in data)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                    $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}, Group = {stuInfo.GroupId}");
            }
        }

        /// <summary>
        /// 获取集合中所有学生数量
        /// </summary>
        [Test]
        public void TestLinq16()
        {
            var students = GetStudentInfos2();
            var studentCount = students.Count(); // 不带条件Count
            var studentCount2 = students.Count(item => item.Age > 18); // 带条件Count

            Console.WriteLine($"All Students Count:{studentCount}");
            Console.WriteLine($"Age Over 18 Students Count:{studentCount2}");
        }

        /// <summary>
        /// 提取学生信息中的名字、性别、年龄 三个数据（使用Select方法）
        /// </summary>
        [Test]
        public void TestLinq17()
        {
            var students = GetStudentInfos2();
            var studentData = students.Select(item => new { item.Name, item.Sex, item.Age }); // Func<TSource, TResult>

            foreach (var item in studentData)
            {
                Console.WriteLine($"Student Name:{item.Name}, Student Sex:{item.Sex}, Student Age:{item.Age}");
            }
        }

        /// <summary>
        /// 提取学生信息中的名字、性别、年龄 三个数据，并根据年龄大小从小到大进行排序
        /// </summary>
        [Test]
        public void TestLinq18()
        {
            var students = GetStudentInfos2();
            var studentData = students.OrderBy(item => item.Age).Select(item => new { item.Name, item.Sex, item.Age }); // Func<TSource, TResult>

            foreach (var item in studentData)
            {
                Console.WriteLine($"Student Name:{item.Name}, Student Sex:{item.Sex}, Student Age:{item.Age}");
            }
        }

        /// <summary>
        /// 提取学生信息中的名字、性别、年龄 三个数据，并根据年龄大小从大到小进行排序（降序排列）
        /// </summary>
        [Test]
        public void TestLinq19()
        {
            var students = GetStudentInfos2();
            var studentData = students.OrderByDescending(item => item.Age).Select(item => new { item.Name, item.Sex, item.Age }); // Func<TSource, TResult>

            foreach (var item in studentData)
            {
                Console.WriteLine($"Student Name:{item.Name}, Student Sex:{item.Sex}, Student Age:{item.Age}");
            }
        }

        /// <summary>
        /// 1. 查询学生信息中的名字、性别、年龄 三个数据，并输出年龄最小的学生信息。
        /// 2. 查询学生信息中位于第一组的第一条记录的学生信息。
        /// </summary>
        [Test]
        public void TestLinq20()
        {
            var students = GetStudentInfos2();
            var student = students.OrderBy(item => item.Age).Select(item => new { item.Name, item.Sex, item.Age, item.GroupId }).First();
            Console.WriteLine($"Student Name:{student.Name}, Student Sex:{student.Sex}, Student Age:{student.Age}, Student Group:{student.GroupId}");

            var student2 = students.First(item => item.GroupId == 1); // 只会查询满足条件的第一条记录（集合首个元素）
            Console.WriteLine($"Student Name:{student2.Name}, Student Sex:{student2.Sex}, Student Age:{student2.Age}, Student Group:{student2.GroupId}");
        }

        [Test]
        public void TestLinq21()
        {
            var students = GetStudentInfos2();

            try
            {
                var student2 = students.First(item => item.GroupId == 10); // 如果没有满足条件的数据，则会抛出InvalidOperationException 异常
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error:{ex.Message}");
            }

            try
            {
                var student2 = students.FirstOrDefault(item => item.GroupId == 10); // 找不到元素时会返回类型的默认值，且不会抛出异常
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error:{ex.Message}");
            }
        }

        [Test]
        public void TestLinq22()
        {
            var students = GetStudentInfos2();

            var maxAgeStudent = students.Max(item => item.Age); // 比较依据：Age
            // 从学生信息的集合中获取年龄最大的学生的年龄
            Console.WriteLine($"Max Student Age:{maxAgeStudent}");

            var minAgeStudent = students.Min(item => item.Age);
            // 从学生信息的集合中获取年龄最小的学生的年龄
            Console.WriteLine($"Min Student Age:{minAgeStudent}");

            var totalChineseScore = students.Sum(item => item.Chinese);
            // 获取学生信息集合中所有学生语文科目的总分
            Console.WriteLine($"Student Chinese Total Score:{totalChineseScore}");

            var avgPhysicsScore = students.Average(item => item.Physics);
            // 获取学生信息集合中所有学生物理科目的平均分
            Console.WriteLine($"Student Physics Average Score:{avgPhysicsScore}");
        }
    }
}
