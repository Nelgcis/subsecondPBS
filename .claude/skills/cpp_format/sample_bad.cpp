// sample_bad.cpp
// 这是一个故意写得不规范的示例文件，用于测试格式化工具

#include <stdlib.h>
#include <string.h>
#include <iostream>

#define MAX_LEN 256
#define SQUARE(a, b) ((a) * (b))

using namespace std;

int g_counter = 0;

class myClass {
	int myValue;
	int* ptr;
public:
	myClass(int v) { myValue = v; }  // 单参数构造未加 explicit

	int getValue() {
		return myValue;
	}

	void doSomething() {
		if (myValue > 0)
			cout << myValue << endl;  // if 没有大括号

		for (int i = 0; i < 10; i++)
			g_counter++;  // for 没有大括号

		// 注释掉的旧代码：
		// int oldResult = myValue * 2;
		// return oldResult;

		int x = 5, y = 10;  // 多变量同行初始化

		int i = 0;
		int result = x + i++;  // 含自增的复合表达式，语义不明确

		// TODO: 需要添加错误处理
		// FIXME: 这里有已知bug

		char* p=NULL;   // NULL 应用 nullptr；指针两侧无空格
		int*q = NULL;

		int status = 8;  // 魔鬼数字
	}
};

int myFunction(int a, int b, int c, int d, int e, int f) {  // 参数超过5个
	int result = (int)a + b;  // C 风格强制转换
	return result;
}

void longFunctionNameThatExceedsColumnLimit(int parameterOne, int parameterTwo, int parameterThree, int parameterFour) {
	// 行宽超过120
}
