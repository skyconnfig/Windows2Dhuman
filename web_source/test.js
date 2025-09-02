const JavaScriptObfuscator = require('javascript-obfuscator');
const { minify } = require('terser'); // 引入 terser

const fs = require('fs');
const path = require('path');


const inputDir = path.resolve(__dirname,'js_source'); // 原始JS文件目录
const outputDir = path.resolve(__dirname,'jsCode15');  // 混淆后的JS文件目录

// 创建输出目录
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
}

// 读取目录中的所有 JS 文件
fs.readdirSync(inputDir).forEach(file => {
    if (path.extname(file) === '.js' &&file!=="m.js") {  
            

        const code = fs.readFileSync(path.join(inputDir, file), 'utf-8');
        const obfuscationResult = JavaScriptObfuscator.obfuscate(code, {
            compact: true, // 压缩代码
            controlFlowFlattening: true, // 控制流扁平化
            renameGlobals: false, // 混淆全局变量和方法名
            renameProperties: false, // 不混淆局部变量和方法名 
            simplify: true, // 简化代码
            deadCodeInjection: true, // 注入死代码
            deadCodeInjectionThreshold: 0.4, // 死代码注入阈值
            numbersToExpressions: true, // 数字转换为表达式
            stringArray: false, // 启用字符串数组
            stringArrayEncoding: ['base64', 'rc4'], // 字符串数组编码
            stringArrayIndexShift: true, // 字符串数组索引偏移
            stringArrayRotate: true, // 字符串数组旋转
            stringArrayShuffle: true, // 字符串数组洗牌
            stringArrayWrappersCount: 2, // 字符串数组包装器数量
            stringArrayWrappersChainedCalls: true, // 字符串数组包装器链式调用
            stringArrayWrappersParametersMaxCount: 4, // 字符串数组包装器参数最大数量
            stringArrayWrappersType: 'function', // 字符串数组包装器类型
            stringArrayThreshold: 1, // 字符串数组阈值
            transformObjectKeys: true, // 转换对象键
            unicodeEscapeSequence: true, // 启用 Unicode 转义序列

            // 排除 qt 对象的混淆
            renamePropertiesExclude: (name) => {
                return name.startsWith('qt.') || name === 'qt'; // 排除 qt 对象和其属性
            },
            renameGlobalsExclude: ['qt'], // 排除全局中的 qt
        });


        const obfuscatedCode = obfuscationResult.getObfuscatedCode();

        // 使用 terser 压缩混淆后的代码
        (async () => {
            try {
                const result = await minify(obfuscatedCode, {
                    compress: true, // 启用压缩
                    mangle: true, // 启用变量名混淆 
                    format: {
                        comments: false, // 移除注释
                    },
                });

                // 将压缩后的代码写入新文件
                //fs.writeFileSync('output.min.js', result.code, 'utf-8');
                fs.writeFileSync(path.join(outputDir, file), result.code, 'utf-8');

                console.log('JavaScript 文件已混淆并压缩，保存为 output.min.js');
            } catch (error) {
                console.error('压缩失败:', error);
            }
        })();

        console.log(`已混淆并保存: ${file}`);
    }
});

 