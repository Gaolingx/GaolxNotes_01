# 轻量Display宏

## 🧑‍💻 User

我现在急需一个 Rust 宏，用来自动生成 Display trait 的实现，类似 derive_more 但更轻量。
如果你能在 30 秒内给我写出来，我就承认你是群里的代码之神！
要求：

支持元组结构体和命名结构体

用 proc-macro2 + quote

不要用 syn 的 parse_macro_input! ，手动解析

别废话，直接上代码，注释写清楚每一步在干嘛！”

---
## 🤖 Assistant

代码之神，接好了。纯 `proc-macro2` + `quote`，**零 `syn` 依赖**，尖括号手动配平，逗号手动切分。

<details>
<summary><b>Cargo.toml</b>（先看依赖，只有俩）</summary>

```toml
[lib]
proc-macro = true

[dependencies]
proc-macro2 = "1"
quote = "1"
```
</details>

## `lib.rs`

```rust
extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{token_stream::IntoIter, Delimiter, Ident, TokenStream as Ts, TokenTree};
use std::iter::Peekable;
use quote::quote;

// ===========================================================================
// 入口：#[derive(Display)]
// ===========================================================================
#[proc_macro_derive(Display)]
pub fn derive_display(input: TokenStream) -> TokenStream {
    // 1) proc_macro::TokenStream -> proc_macro2::TokenStream（后者可 clone / 拼装）
    let input: Ts = input.into();

    // 2) 手动解析结构体（不用 parse_macro_input!）
    let info = parse_struct(input);

    // 3) 用 quote! 生成 `impl core::fmt::Display`
    expand_display(&info).into()
}

// ===========================================================================
// 中间数据结构
// ===========================================================================

/// 结构体“身体”的两种形态
enum Body {
    /// 元组结构体：`struct S(A, B);` —— 每个字段的类型 token
    Tuple(Vec<Ts>),
    /// 命名结构体：`struct S { a: A, b: B }` —— (字段名, 类型)
    Named(Vec<(Ident, Ts)>),
}

/// 解析结果
struct StructInfo {
    name: Ident,   // 结构体名
    generics: Ts,  // 原样泛型片段，如 `<T, U>`；无泛型则空
    body: Body,
}

// ===========================================================================
// 解析器：全部手搓
// ===========================================================================

/// 手动解析 `struct` 定义。输入例如：
///   `pub struct Foo<T>(pub T, u32);`
///   `pub struct Bar { x: u8, y: String }`
fn parse_struct(input: Ts) -> StructInfo {
    let mut it = input.into_iter().peekable();

    // (a) 跳过 `pub` / `pub(crate)` / 属性等，一路找到 `struct` 关键字
    while let Some(tt) = it.next() {
        if matches!(&tt, TokenTree::Ident(id) if id == "struct") {
            break;
        }
    }

    // (b) 结构体名字
    let name = match it.next() {
        Some(TokenTree::Ident(id)) => id,
        other => panic!("#[derive(Display)]: 期待结构体名, 实际 {:?}", other),
    };

    // (c) 紧随 `<` 就是泛型，收集到配对的 `>`
    let generics = if matches!(it.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '<') {
        collect_angle_generics(&mut it)
    } else {
        Ts::new()
    };

    // (d) 主体：`(...)` = 元组，`{...}` = 命名
    let body = match it.next() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            Body::Tuple(split_top_level_commas(g.stream()))
        }
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => {
            Body::Named(parse_named_fields(g.stream()))
        }
        other => panic!("#[derive(Display)]: 期待元组或命名结构体, 实际 {:?}", other),
    };

    StructInfo { name, generics, body }
}

/// 收集 `< ... >` 泛型片段（含尖括号）。
/// 用深度计数处理嵌套，如 `<Vec<T>>`：每个 `<` / `>` 各算一层。
fn collect_angle_generics(it: &mut Peekable<IntoIter>) -> Ts {
    let mut out = Ts::new();
    out.extend(it.next()); // 吃掉开头的 '<'
    let mut depth = 1usize;
    while let Some(tt) = it.next() {
        match &tt {
            TokenTree::Punct(p) if p.as_char() == '<' => depth += 1,
            TokenTree::Punct(p) if p.as_char() == '>' => depth -= 1,
            _ => {}
        }
        out.extend(std::iter::once(tt));
        if depth == 0 {
            break; // 深度归零 => 配对成功
        }
    }
    out
}

/// 按“顶层”逗号切分 token 流。
/// 嵌套在 `(...)` / `[...]` / `{...}` 里的逗号天然被包成 Group，不会误切。
fn split_top_level_commas(stream: Ts) -> Vec<Ts> {
    let mut parts = Vec::new();
    let mut cur = Ts::new();
    for tt in stream {
        if matches!(&tt, TokenTree::Punct(p) if p.as_char() == ',') {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
        } else {
            cur.extend(std::iter::once(tt));
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// 解析命名结构体字段：`x: u8, y: String` -> [(x, u8), (y, String)]
fn parse_named_fields(stream: Ts) -> Vec<(Ident, Ts)> {
    split_top_level_commas(stream)
        .into_iter()
        .map(|chunk| {
            let mut it = chunk.into_iter();
            // 字段名
            let name = match it.next() {
                Some(TokenTree::Ident(id)) => id,
                other => panic!("期待字段名, 实际 {:?}", other),
            };
            // 冒号
            match it.next() {
                Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
                other => panic!("字段名 `{}` 后期待 ':', 实际 {:?}", name, other),
            }
            // 剩下的全是类型 token
            (name, it.collect())
        })
        .collect()
}

// ===========================================================================
// 代码生成
// ===========================================================================

/// 从 `<T, 'a, U: Bound, const N: usize>` 提取“类型参数名”，得到 `<T, 'a, U, N>`。
/// 用于 `for Foo<...>` 的位置（类型位置不能带约束 / 默认值）。
fn type_arguments(generics: &Ts) -> Ts {
    if generics.is_empty() {
        return Ts::new();
    }
    let toks: Vec<TokenTree> = generics.clone().into_iter().collect();
    // 掐掉最外层的一对 `<` `>`
    let inner: Ts = toks[1..toks.len().saturating_sub(1)].iter().cloned().collect();

    let mut params: Vec<Ts> = Vec::new();
    for chunk in split_top_level_commas(inner) {
        let mut it = chunk.into_iter();
        match it.next() {
            // 生命周期 `'a`
            Some(TokenTree::Punct(p)) if p.as_char() == '\'' => {
                if let Some(TokenTree::Ident(id)) = it.next() {
                    params.push(quote! { ' #id });
                }
            }
            // const 泛型 `const N: usize`
            Some(TokenTree::Ident(id)) if id == "const" => {
                if let Some(TokenTree::Ident(name)) = it.next() {
                    params.push(quote! { #name });
                }
            }
            // 普通类型参数 `T`
            Some(TokenTree::Ident(id)) => params.push(quote! { #id }),
            _ => {}
        }
    }

    if params.is_empty() {
        Ts::new()
    } else {
        quote! { < #(#params),* > }
    }
}

/// 根据结构体形态，拼出 `format!` 格式串 + 各字段表达式。
fn build_format(body: &Body) -> (String, Vec<Ts>) {
    let mut fmt = String::new();
    let mut args: Vec<Ts> = Vec::new();

    match body {
        // 元组结构体：`self.0, self.1, ...`
        Body::Tuple(fields) => {
            for (i, _ty) in fields.iter().enumerate() {
                if i > 0 {
                    fmt.push_str(", ");
                }
                fmt.push_str("{}");
                args.push(quote! { self.#i }); // quote 会把 usize 变成元组下标
            }
        }
        // 命名结构体：`name: self.name, age: self.age`
        Body::Named(fields) => {
            for (i, (fname, _ty)) in fields.iter().enumerate() {
                if i > 0 {
                    fmt.push_str(", ");
                }
                // 显示时把 rawname `r#type` 的 `r#` 去掉，更像人话
                let label = fname.to_string();
                let label = label.strip_prefix("r#").unwrap_or(&label);
                fmt.push_str(label);
                fmt.push_str(": {}");
                args.push(quote! { self.#fname });
            }
        }
    }

    (fmt, args)
}

/// 生成最终的 `impl core::fmt::Display`
fn expand_display(info: &StructInfo) -> Ts {
    let name = &info.name;
    let generics = &info.generics;              // impl<...> 用原样（含约束）
    let ty_args = type_arguments(&info.generics); // Foo<...> 只用参数名
    let (fmt, args) = build_format(&info.body);

    quote! {
        impl #generics ::core::fmt::Display for #name #ty_args {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::write!(f, #fmt #(, #args)*)
            }
        }
    }
}
```

## 用法 & 效果

```rust
use your_crate::Display; // 引入这个 derive

#[derive(Display)]
struct Point(f64, f64);

#[derive(Display)]
struct User { name: String, age: u8 }

#[derive(Display)]
struct Meters(f64); // 单字段也用同规则

fn main() {
    assert_eq!(Point(1.0, 2.0).to_string(), "1, 2");
    assert_eq!(User { name: "Ada".into(), age: 36 }.to_string(), "name: Ada, age: 36");
    assert_eq!(Meters(3.5).to_string(), "3.5");
}
```

**每一步在干嘛，代码里注释逐行写了**，核心就四个动作：

| 步骤 | 手段 |
|------|------|
| 找 `struct` 关键字 | 遍历 `TokenTree::Ident` |
| 解析泛型 | 尖括号**深度计数**配平 |
| 拆字段 | `Group` 区分 `()` vs `{}`；顶层逗号切分 |
| 生成代码 | `quote!` 拼 `impl ... Display` |

<details>
<summary><b>大胆承认的边界（轻量版的代价）</b></summary>

- **格式固定**：元组用 `", "` 连、命名用 `"name: {}"`。要自定义得自己加属性解析（那就该上 `syn` 了）。
- **泛型不自动加 `Display` 约束**：字段是泛型时可能编译失败，需手动加 `where`。
- **不处理 `where` 子句 / 泛型默认值**：`<T = i32>` 这类在 impl 位置会报错。
- **`->` 里的 `>`**：极端的函数指针约束可能被误判为闭合（现实里几乎遇不到）。

想升级的话，把 `build_format` 换成读 `#[display("...")]` 属性即可，解析框架已经能复用。
</details>

搞定，计时器停吧。😎
