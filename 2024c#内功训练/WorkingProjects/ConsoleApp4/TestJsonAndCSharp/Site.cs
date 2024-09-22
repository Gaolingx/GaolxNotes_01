
namespace TestJsonAndCSharp
{
    public class Site
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Url { get; set; }
        public List<SiteType> SiteTypes { get; set; }
    }
}
