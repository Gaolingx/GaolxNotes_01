﻿using NUnit.Framework;

namespace TestLinq
{
    internal class StudyLinq
    {
        #region 测试数据
        public static List<StuInfo> GetStudentInfos()
        {
            List<StuInfo> stuInfos = new List<StuInfo>()
            {
                new StuInfo { Id = 1001, Name = "流萤", Sex = "女", Age = 20, Chinese = 100, Math = 120, English = 95, Physics = 70, Score = 500, Grade = "A" },
                new StuInfo { Id = 1002, Name = "符玄", Sex = "女", Age = 20, Chinese = 105, Math = 130, English = 100, Physics = 80, Score = 500, Grade = "A" },
                new StuInfo { Id = 1003, Name = "爱莉希雅", Sex = "女", Age = 18, Chinese = 110, Math = 90, English = 105, Physics = 65, Score = 500, Grade = "B" },
                new StuInfo { Id = 1003, Name = "琪亚娜", Sex = "女", Age = 19, Chinese = 90, Math = 85, English = 100, Physics = 60, Score = 500, Grade = "B" }
            };
            return stuInfos;
        }

        public static List<ClassGroup> GetClassGroups()
        {
            List<ClassGroup> classGroups = new List<ClassGroup>()
            {
                new ClassGroup{Id = 1001,GroupName="组1"},
                new ClassGroup{Id = 1002,GroupName="组2"},
                new ClassGroup{Id = 1003,GroupName="组3"},
                new ClassGroup{Id = 1004,GroupName="组4"},
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
                             where p.Id == g.Id
                             select new { p.Name, g.GroupName };

            foreach (var item in stuInfoLst)
            {
                Console.WriteLine($"Name = {item.Name}, GroupName = {item.GroupName}");
            }
        }
    }
}
