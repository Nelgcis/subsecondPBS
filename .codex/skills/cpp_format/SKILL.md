---
name: cpp_format
description: >
  将 C++/.h 源文件格式化为符合公司医疗器械软件 C++ 编码规范的标准格式。
  涵盖命名（4.1）、格式（4.2）、注释（4.3）、头文件（4.4）、作用域（4.5）、
  类（4.6）、函数（4.7）、特性（4.8）全部条款。
  只修改格式，不改变任何逻辑实现；潜在含义变化记入 format_changes.log 由用户确认。
  调用方式：python3 cpp_formatter.py <file_or_dir> [--inplace] [--log <path>]
usage: >
  当用户要求"按规范格式化 C++ 代码"、"检查编码规范"、"格式化 .cpp/.h 文件"时触发。
  先阅读本 SKILL.md 了解规范全文，再调用 cpp_formatter.py 执行格式化，
  最后将 format_changes.log 中的 [WARN] 条目呈现给用户确认。
tools:
  - cpp_formatter.py
---

# C++ 代码规范格式化 Skill

## 概述

本 Skill 用于将 C++ 代码文件（或代码片段）格式化为符合公司医疗器械软件 C++ 编码规范（GMP-C-02-14 或同类规范文档）的标准格式。

**核心原则：只修改格式，不改变任何逻辑和实现细节。** 若存在潜在含义变化，必须写入 `format_changes.log` 并由用户确认。

---

## 使用方式

```
输入：一个或多个 .cpp / .h 文件路径，或代码字符串
输出：
  - 格式化后的文件（原文件同目录，带 .formatted 后缀；或原位覆盖，取决于用户选择）
  - format_changes.log（在执行目录下，记录所有变更及潜在风险项）
```

调用示例（在 Claude 对话中）：
- "帮我按规范格式化这个文件：src/MyClass.cpp"
- "按照编码规范检查并修正这段代码"
- "格式化 include/ 下所有头文件"

---

## 规范全文速查（来源：公司 C++ 编码规范）

### 4.1 命名规范

| 元素 | 风格 | 说明 |
|------|------|------|
| 类、结构体、枚举、联合体、作用域、函数 | 大驼峰 UpperCamelCase | 首字母大写 |
| 全局变量、命名空间变量、类静态变量、局部变量、函数参数、成员变量 | 小驼峰 lowerCamelCase | 首字母小写 |
| 宏、常量(const)、枚举值、goto 标签 | UPPER_CASE_WITH_UNDERSCORES | 全大写下划线分隔 |
| C++ 文件 | `.cpp` 结尾 | 文件名与类名保持一致 |
| 头文件 | `.h` 结尾 | 文件名与类名保持一致 |
| 函数命名 | 大驼峰，动词或动宾结构 | 如 `AddElement`、`GetElement`、`IsEmpty` |
| 全局变量 | 加 `g_` 前缀 | 静态变量无需特殊前缀 |

**禁止**：除非有明确必要性，否则不用 `typedef`/`#define` 对基本数据类型重定义。

---

### 4.2 格式

#### 行宽
- 每行不超过 **120 个字符**；超过时需换行。

#### 缩进
- 只使用**空格**缩进，每次缩进 **4 个空格**。
- 禁止使用 Tab。

#### 大括号
- 函数（不含 lambda）：左大括号**另起一行放行首**，独占一行。
- 其他（struct、class、if、for 等）：左大括号**跟随语句放行末**，前置 1 个空格。
- 右大括号独占一行，除非后跟 `else`/`else if`、`while`（do-while）、逗号、分号。

```cpp
// 结构体：左括号跟行末
struct MyType { // 跟随语句放行末，前置1空格
    ...
};

// 函数：左括号独占一行放行首
int Foo(int a)
{ // 函数左大括号独占一行，放行首
    if (...) {
        ...
    } else {
        ...
    }
}

// 空函数体：括号可同行
class MyClass {
public:
    MyClass() : value_(0) {}
private:
    int value_;
};
```

#### 函数声明/定义换行
- 返回值类型与函数名同行。
- 参数列表放一行；超出行宽时换行并对齐，左括号总跟函数名，右括号总跟最后参数。

```cpp
ReturnType FunctionName(ArgType paramName1, ArgType paramName2) // 好：全在同一行
{
    ...
}

// 行宽不满足所有参数，进行换行
ReturnType VeryVeryVeryLongFunctionName(ArgType paramName1,
                                        ArgType paramName2, // 和上一行参数对齐
                                        ArgType paramName3)
{
    ...
}

// 行宽限制，4空格缩进对齐
ReturnType LongFunctionName(ArgType paramName1, ArgType paramName2, ArgType
    paramName3, ArgType paramName4, ArgType paramName5) // 4空格缩进
{
    ...
}

// 行宽不满足第1个参数，直接换行
ReturnType ReallyReallyReallyReallyLongFunctionName( // 行宽不满足第1个参数，直接换行
    ArgType paramName1, ArgType paramName2, ArgType paramName3) // 4空格缩进
{
    ...
}
```

#### 函数调用换行
- 参数列表放一行；超出时换行对齐，左括号跟函数名，右括号跟最后参数。

```cpp
ReturnType result = FunctionName(paramName1, paramName2); // 好：参数放一行

ReturnType result = FunctionName(paramName1,
                                 paramName2, // 保持与上方参数对齐
                                 paramName3);

ReturnType result = FunctionName(paramName1, paramName2,
    paramName3, paramName4, paramName5); // 4空格缩进

ReturnType result = VeryVeryVeryLongFunctionName( // 行宽不满足第1个参数，直接换行
    paramName1, paramName2, paramName3); // 4空格缩进
```

#### 控制语句
- `if` 语句**必须**使用大括号，即便只有一条语句。
- `for`/`while` 循环**必须**加大括号，即便循环体为空或只有一条语句。
- `switch` 的 `case`/`default` 缩进一层。

```cpp
switch (var) {
    case 0: // 好：缩进
        DoSomething1(); // 好：缩进
        break;
    case 1: { // 好：带大括号格式
        DoSomething2();
        break;
    }
    default:
        break;
}
```

#### 长表达式换行
- 在较低优先级运算符/连接符**后面**截断，运算符放行末（表示"未结束"）。
- 换行后保持合理对齐，或 4 空格缩进。

```cpp
if (currentValue > threshold && // 好：换行后，逻辑操作符放在行尾
    someCondition) {
    DoSomething();
}

int sum = longVaribleName1 + longVaribleName2 + longVaribleName3 +
    longVaribleName4 + longVaribleName5 + longVaribleName6; // 好：4空格缩进

int sum = longVaribleName1 + longVaribleName2 + longVaribleName3 +
          longVaribleName4 + longVaribleName5 + longVaribleName6; // 好：保持对齐
```

#### 变量初始化
- 每行只有一个变量初始化语句。
- 结构体/数组初始化换行时保持 4 空格缩进。

```cpp
int maxCount = 10;
bool isCompleted = false;

const int rank[] = {
    16, 16, 16, 16, 32, 32, 32, 32,
    64, 64, 64, 64, 32, 32, 32, 32
};
```

#### 指针与引用命名
- `*` 靠左或靠右均可，但不能两边都有空格，也不能两边都没空格。

```cpp
int* p = NULL;  // 好
int *p = NULL;  // 好
int*p = NULL;   // 坏
int * p = NULL; // 坏

// const 修饰时，* 不跟随变量也不跟随类型（居中）
const char* const VERSION = "V100";
```

- `&` 同理，靠左或靠右均可，两边都有/都无空格为坏。

```cpp
int i = 8;
int& p = i;  // 好
int &p = i;  // 好
int & p = i; // 坏
int&p = i;   // 坏

int*& rp = pi;  // 好，指针的引用，*& 一起跟随类型
int *&rp = pi;  // 好，指针的引用，*& 一起跟随变量名
int* &rp = pi;  // 好，指针的引用，* 跟随类型，& 跟随变量名
```

#### 编译预处理
- `#` 统一放在行首，即使预处理代码嵌入函数体内。
- 内嵌预处理语句 `#` 可按缩进要求对齐，区分层次。

```cpp
#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16) // "#"放在行首
#define ATOMIC_X86_HAS_CMPXCHG16B 1 // 好："#"放在行首
#else
#define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif

// 内嵌，区分层次
#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
    #define ATOMIC_X86_HAS_CMPXCHG16B 1 // 好：区分层次，便于阅读
#else
    #define ATOMIC_X86_HAS_CMPXCHG16B 0
#endif
```

---

### 4.3 注释

- 尽量通过清晰的架构逻辑和好的命名提高可读性，需要时才加注释。
- 注释内容简洁、明了、无二义性，不冗余。

#### 函数头注释（放在声明或定义上方）

```cpp
// 使用 // 风格
// 单行函数头
int Func1(void);

// 多行函数头
// 第二行
int Func2(void);

// 使用 /* */ 风格
/* 单行函数头 */
int Func1(void);

/*
 * 多行函数头
 * 第二行
 */
int Func3(void);
```

- 函数尽量通过函数名自注释，按需写函数头注释。
- **不要**写无用、信息冗余的函数头；**不要**写空有格式的函数头。
- 函数头注释内容可包含：功能说明、返回值、性能约束、用法、内存约定、算法实现、可重入要求等。

```cpp
// 好的例子：
/*
 * 返回实际写入的字节数，-1 表示写入失败
 * 注意，内存 buf 由调用者负责释放
 */
int WriteString(const char *buf, int len);

// 坏的例子（空有格式，无实质内容）：
/*
 * 函数名：WriteString
 * 功能：写入字符串
 * 参数：
 * 返回值：
 */
int WriteString(const char *buf, int len);
```

#### 代码注释
- 放于对应代码的上方或右边。
- 使用 `//` 或 `/* */` 均可。

```cpp
// 这是单行注释
DoSomething();

// 这是多行注释
// 第二行
DoSomething();

int foo = 100; // 放右边的注释

/* 这是单行注释 */
DoSomething();

/*
 * 另一种方式的多行注释
 */
DoSomething();

int bar = 200; /* 放右边的注释 */
const int A_CONST = 100; /* 相关的同类注释，可以考虑上下对齐 */
const int ANOTHER_CONST = 200; /* 与左侧代码保持间隔 */
```

#### 其他注释规则
- **不用**的代码段直接删除，不要注释掉。
- 正式交付给客户的代码**不能**包含 `TODO`/`TBD`/`FIXME` 注释。

---

### 4.4 头文件

- 每个 `.cpp` 文件应有对应的 `.h` 文件，禁止头文件循环依赖。
- 头文件必须编写 `#define` 保护，防止重复包含。

```cpp
#ifndef TIMER_INCLUDE_TIMER_H
#define TIMER_INCLUDE_TIMER_H
...
#endif
```

- 禁止通过 `extern` 声明方式引用外部函数接口、变量。
- 禁止在 `extern "C"` 中包含头文件。

---

### 4.5 作用域

- cpp 文件中不需要导出的变量/常量/函数，使用匿名 `namespace` 封装。
- 不要在头文件及源文件 `#include` 之前使用 `using` 导入命名空间。
- 优先使用命名空间管理全局函数；与某个 class 直接相关时可使用静态成员函数。
- 尽量避免使用全局变量，考虑使用单例模式。

---

### 4.6 类

- 成员变量必须显式初始化（例外：有默认构造函数时可省略）。
- 优先使用声明时初始化（C++11）和构造函数初始化列表。
- 单参数构造函数声明为 `explicit`，防止隐式转换。

```cpp
class Foo {
public:
    explicit Foo(const string& name) : name_(name)
    {
    }
private:
    string name_;
};
```

- 不需要的拷贝构造/赋值操作符/移动构造/赋值操作符，明确禁止（设为 private 或使用 `= delete`）。
- 拷贝构造和拷贝赋值操作符成对出现或成对禁止。
- 移动构造和移动赋值操作符成对出现或成对禁止。
- **禁止**在构造函数和析构函数中调用虚函数。
- 基类析构函数应声明为 `virtual`。
- **禁止**虚函数使用缺省参数值。
- **禁止**重新定义继承而来的非虚函数。

---

### 4.7 函数

- 函数不超过 **50 行**（不含空行和注释）；算法函数例外。
- 内联函数建议不超过 **10 行**；虚函数、递归函数不能做内联函数。
- 函数参数使用**引用**取代指针（例外：编译期长度未知的数组可用指针）。
- 使用强类型参数，避免使用 `void*`。
- 函数参数个数建议不超过 **5 个**。

---

### 4.8 特性

#### 4.8.1 常量与初始化

- **禁止**用宏表示常量，使用 `const` 或 `constexpr`。

```cpp
#define MAX_MSISDN_LEN 20 // 不好

const int MAX_MSISDN_LEN = 20; // 好

constexpr int MAX_MSISDN_LEN = 20; // 好（C++11）
```

- 一组相关整型常量定义为枚举（`enum`）。
- 枚举值需对应具体数值时，声明时显式赋值；仅用于分类时不显式赋值。
- 尽量避免枚举值重复；必须重复时用已定义的枚举修饰。
- **禁止**使用魔鬼数字（看不懂、难以理解的数字）；多处使用的数字必须定义 `const` 常量。
- 禁止：`const int ZERO = 0`（符号命名限制了取值）；禁止：`const int XX_TIMER_INTERVAL_300MS = 300`（应直接用 `XX_TIMER_INTERVAL_MS`）。
- 一个常量只表示一个特定功能，不能有多种用途。

#### 4.8.2 表达式

- 变量使用时才声明并初始化。
- 含自增/自减运算的表达式中，**禁止**再次引用该变量（自增/自减单独放一行）。

```cpp
x = b[i] + i;
i++; // 好：单独一行

// 函数参数同理
i++;       // 好：单独一行
x = Func(i, i);
```

- `switch` 语句要有 `default` 分支。
- 表达式比较遵循**左侧倾向于变化、右侧倾向于不变**原则：

```cpp
if (value == MAX) { ... }   // 好
if (value < MAX) { ... }    // 好
// 区间描述时前半段可以常量在左：
if (MIN < value && value < MAX) { ... }
```

- 使用括号明确操作符优先级（操作符不同时需加括号）。

#### 4.8.3 类型转换

- 使用 C++ 风格类型转换，禁止 C 风格转换。
  - `dynamic_cast`：继承体系下行转换（尽量避免）。
  - `static_cast`：值强制转换或上行转换（相对安全）。
  - `reinterpret_cast`：不相关类型转换（尽可能少用）。
  - `const_cast`：移除 const 属性（尽可能少用）。
  - 算数转换（无类型信息丢失，如 float→double）：推荐大括号初始化方式。

```cpp
double d{ someFloat };
int64_t i{ someInt32 };
```

#### 4.8.4 资源分配和释放

- 单个对象释放用 `delete`，数组对象释放用 `delete[]`。
- 使用 RAII 特性追踪动态分配：构造时获取资源，析构时释放资源。

#### 4.8.5 标准库

- 不要保存 `std::string` 的 `c_str()` 返回的指针（生命周期不保证）。
- 使用 `std::string` 代替 `char*`。
- **禁止**使用 `auto_ptr`，用 `std::unique_ptr` 代替。
- 使用 C++ 标准头文件时使用 `<cstdlib>` 而非 `<stdlib.h>`。

#### 4.8.6 Const 用法

- 指针/引用类型形参不需要修改时，使用 `const`。
- 不修改成员变量的成员函数用 `const` 修饰。

```cpp
int PrintValue() const // const 修饰成员函数
{
    std::cout << value_ << std::endl;
}
int GetValue() const
{
    return value_;
}
```

- 初始化后不再修改的成员变量定义为 `const`。

#### 4.8.7 异常

- 不会抛出异常的函数声明为 `noexcept`。
- 默认构造函数、析构函数、`swap` 函数、move 操作符都不应抛出异常。

#### 4.8.8 模板

- 模板编程最好只用在少量基础组件/数据结构上，复杂度最小化，不对外暴露。
- 在实现里使用模板，给用户暴露的接口不使用模板。
- 模板代码上写尽可能详细的注释。

```cpp
// 不推荐宏函数
#define SQUARE(a, b) ((a) * (b))

// 推荐模板函数
template<typename T> T Square(T a, T b) { return a * b; }
```

#### 4.8.9 宏

- 尽可能少使用复杂的宏；用模板函数、内联函数替换宏函数。

#### 4.8.10 代码简洁性和安全性

- 需要时使用明确的类型。
- `auto` 只用于局部变量。
- 重写虚函数时使用 `override` 或 `final` 关键字。
- 使用 `delete` 关键字删除函数（而非声明为 `private` 但不实现）。
- 使用 `nullptr` 而非 `NULL` 或 `0`。
- 使用 `using` 而非 `typedef`。
- **禁止** `std::move` 操作 `const` 对象。

#### 4.8.11 智能指针

- 优先使用智能指针而非原始指针管理资源（性能敏感/兼容性场景例外）。
- 优先使用 `unique_ptr` 而非 `shared_ptr`。
- 使用 `std::make_unique` 而非 `new` 创建 `unique_ptr`（需自定义 deleter 时例外）。
- 使用 `std::make_shared` 而非 `new` 创建 `shared_ptr`（需自定义 deleter 时例外）。

#### 4.8.12 Lambda

- 仅在函数无法工作时（需捕获局部变量或写局部函数）使用 lambda。
- 非局部范围使用 lambda 时，避免按引用捕获（防止悬空引用）。
- 捕获 `this` 时，显式捕获所有变量。
- 避免使用默认捕获模式（`[=]` 和 `[&]`），明确写出需要捕获的变量。

```cpp
// 不好：默认按值捕获，静态变量实际未被复制
return [=]() {
    ++baseValue; // 修改会影响静态变量
    return baseValue + addend;
};

// 好：使用 C++14 捕获初始化，明确拷贝
return [addend, baseValue = baseValue]() mutable {
    ++baseValue; // 修改自己的拷贝，不影响静态变量
    return baseValue + addend;
};
```

#### 4.8.13 接口

- 不涉及所有权时，用 `T*` 或 `T&` 作参数，而非智能指针。
- 只在需要明确所有权机制时，通过智能指针转移或共享所有权。

---

## 执行流程（Claude 执行本 Skill 时遵循此流程）

### Step 1：读取源文件
读取待格式化的 `.cpp`/`.h` 文件，逐行分析。

### Step 2：逐项检查并记录变更
按照上述规范条目逐项检查，对每个需要修改的地方：
- 若为**纯格式变更**（缩进、空格、换行、大括号位置）：直接修改，记入 log（INFO 级别）。
- 若存在**潜在含义变化**（如：去掉注释掉的代码、修改多行宏展开方式、调整自增/自减顺序等）：记入 log（WARN 级别），**不自动修改**，等待用户确认。

### Step 3：生成 format_changes.log

日志格式：
```
[INFO]  文件名:行号  原内容 → 新内容  (规范条款 X.X)
[WARN]  文件名:行号  描述潜在含义变化风险，原内容 → 建议内容  (规范条款 X.X)
[SKIP]  文件名:行号  无法自动修复，需人工处理  (规范条款 X.X)
```

### Step 4：输出格式化后的文件
- 将纯格式变更应用后的文件输出。
- 所有 `[WARN]` 条目对应位置保留原样，并在该行前加注释 `// FORMAT-WARN: 见 format_changes.log 第N行`。

### Step 5：提示用户
告知用户：
1. 已完成格式化，输出文件路径。
2. 共计 INFO 条数（已自动应用）和 WARN 条数（需用户确认）。
3. 请查看 `format_changes.log` 并决定 WARN 项是否保留。

---

## 潜在含义变化的判断标准（WARN 触发条件）

| 情形 | 是否 WARN |
|------|-----------|
| 修改缩进（空格数量/Tab→空格） | INFO（纯格式） |
| 移动大括号位置 | INFO（纯格式） |
| 拆分多变量初始化为单行 | INFO（纯格式） |
| 删除注释掉的代码块 | WARN（可能有意保留） |
| 删除 TODO/TBD/FIXME 注释 | WARN（可能未完成工作） |
| 修改宏展开方式 | WARN（展开顺序可能影响行为） |
| 修改自增/自减表达式顺序 | WARN（执行顺序改变） |
| 修改指针/引用的 * & 位置（影响 const 语义） | WARN（如 `const int* p` vs `int* const p`） |
| 修改函数参数换行对齐（不影响逻辑） | INFO（纯格式） |
| 修改注释内容（非格式） | SKIP（不修改注释内容） |
| 修改命名（命名规范违规） | WARN（涉及接口/符号变更） |

---

## 注意事项

1. **绝不修改代码逻辑**：只改格式，不改算法、不改变量值、不改函数签名语义。
2. **命名规范违规仅报告**：命名不符合规范时，记入 `[WARN]` 提示，不自动重命名（重命名会影响所有调用方）。
3. **TODO/TBD/FIXME**：记入 `[WARN]`，提示用户手动决定是保留（开发中）还是删除（正式交付前）。
4. **注释掉的代码**：记入 `[WARN]`，提示用户决定是删除还是保留。
5. **宏函数**：记入 `[WARN]`，建议替换为模板/内联函数，但不自动修改（可能有平台兼容性原因）。
6. **多文件处理**：若输入为目录，递归处理所有 `.cpp`/`.h` 文件，所有变更汇总到同一个 `format_changes.log`。
