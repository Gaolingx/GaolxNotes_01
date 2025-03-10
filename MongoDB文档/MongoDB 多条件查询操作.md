## 基本查询

```javascript
db.users.find({ age: 25 })
```

## **AND 查询 （多个条件同时满足）**

```javascript
db.users.find({ age: 25, name: "Fantasy1" })
```

## **OR 查询（多个条件满足其中之一）**

```javascript
// 查询年龄等于25或名字为Fantasy1的数据
db.users.find({ 
   $or: [ 
	 { age: 25 }, 
	 { name: "Fantasy1" } 
   ] 
})
// 查询年龄小于25或名字为Fantasy10的数据
db.users.find({ 
   $or: [ 
	 { age: { $lt: 25 } }, 
	 { name: "Fantasy1" } 
   ] 
})
```

## **组合查询（AND + OR）**

```javascript
// 查询年龄大于等于25，name里有Fantasy1或Fantasy2的数据
db.users.find({ 
  age: { $gte: 25 }, 
  $or: [ 
    { name: "Fantasy1" }, 
    { name: "Fantasy2" } 
  ] 
})
```