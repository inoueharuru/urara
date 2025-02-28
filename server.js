// 引入必要的Node.js模块
let http = require("http");
let fs = require("fs");
let path = require("path");
let url = require("url");
let querystring = require("querystring");

// 用于处理文件上传的依赖库（实际应用中需要安装这些npm包）
// 这里只是模拟，真实情况请使用npm安装：
// npm install --save multer sharp tensorflow-node@2.8.0

// 创建模型加载和预测的模拟函数
// 实际应用中，这些应由TensorFlow.js Node.js API实现
const models = {
  ct: null,
  pathology: null,
  fusion: null
};

// 模拟模型加载
async function loadModels() {
  console.log("加载疾病分类模型...");
  try {
    // 实际应用中应使用 tf.loadLayersModel 从文件系统加载模型
    // models.ct = await tf.node.loadLayersModel(`file://${__dirname}/models/ct-model/model.json`);
    // models.pathology = await tf.node.loadLayersModel(`file://${__dirname}/models/pathology-model/model.json`);
    // models.fusion = await tf.node.loadLayersModel(`file://${__dirname}/models/fusion-model/model.json`);
    console.log("所有模型加载完成");
    return true;
  } catch (error) {
    console.error("模型加载失败:", error);
    return false;
  }
}

// 处理上传的CT文件
async function processCTImage(imageBuffer) {
  try {
    console.log("处理CT图像...");
    // 实际应用中应使用sharp或其他图像处理库处理图像
    // const image = await sharp(imageBuffer)
    //   .resize(224, 224)
    //   .toBuffer();
    // 
    // const tensor = tf.node.decodeImage(image)
    //   .toFloat()
    //   .div(255.0)
    //   .expandDims();
    // return tensor;
    
    // 模拟返回CT特征，实际应由模型生成
    return {
      features: [0.32, 0.45, 0.67, 0.23],
      description: "CT图像显示病变区域密度不均，边界清晰，内部可见钙化灶"
    };
  } catch (error) {
    console.error("CT图像处理失败:", error);
    throw error;
  }
}

// 处理上传的病理图像
async function processPathologyImage(imageBuffer) {
  try {
    console.log("处理病理图像...");
    // 实际应用中应使用sharp或其他图像处理库处理图像
    // const image = await sharp(imageBuffer)
    //   .resize(224, 224)
    //   .toBuffer();
    // 
    // const tensor = tf.node.decodeImage(image)
    //   .toFloat()
    //   .div(255.0)
    //   .expandDims();
    // return tensor;
    
    // 模拟返回病理特征，实际应由模型生成
    return {
      features: [0.56, 0.78, 0.34, 0.91],
      description: "病理切片显示梭形细胞排列成束状，可见钙化灶和骨样组织"
    };
  } catch (error) {
    console.error("病理图像处理失败:", error);
    throw error;
  }
}

// 融合特征并进行疾病预测
async function predictDisease(ctFeatures, pathologyFeatures) {
  try {
    console.log("融合特征并预测疾病...");
    // 实际应用中应使用融合模型进行预测
    // const concatenatedFeatures = tf.concat([ctFeatures, pathologyFeatures], 1);
    // const prediction = models.fusion.predict(concatenatedFeatures);
    // const probability = prediction.dataSync()[0];
    
    // 模拟预测结果，实际应由融合模型生成
    // 随机概率，模拟预测结果
    const probability = 0.78;
    
    let diagnosis;
    let analysisText;
    
    if (probability > 0.5) {
      diagnosis = "骨化纤维瘤";
      analysisText = "基于CT和病理图像的多模态分析显示，该病例更符合骨化纤维瘤的特征。CT显示边界清晰的溶骨性病变，病理学特征与骨化纤维瘤相符。";
    } else {
      diagnosis = "骨纤维异常增殖综合征";
      analysisText = "基于CT和病理图像的多模态分析显示，该病例更符合骨纤维异常增殖综合征的特征。CT显示病变边界不清，病理学特征与骨纤维异常增殖综合征相符。";
    }
    
    return {
      diagnosis: diagnosis,
      probability: probability,
      ctFeatures: ctFeatures.description,
      pathologyFeatures: pathologyFeatures.description,
      combinedAnalysis: analysisText
    };
  } catch (error) {
    console.error("疾病预测失败:", error);
    throw error;
  }
}

// 处理文件上传和分析请求
async function handleFileUpload(req, res) {
  try {
    console.log("接收文件上传请求...");
    
    // 读取请求体数据
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', async () => {
      try {
        // 解析multipart/form-data请求体
        // 注意：实际应用中应使用multer等库处理文件上传
        // 这里仅作为简化示例
        
        // 模拟处理上传的文件
        console.log("接收到文件上传，开始处理...");
        
        // 模拟处理CT和病理图像
        const ctFeatures = await processCTImage(Buffer.from("模拟CT图像数据"));
        const pathologyFeatures = await processPathologyImage(Buffer.from("模拟病理图像数据"));
        
        // 融合特征并预测
        const result = await predictDisease(ctFeatures, pathologyFeatures);
        
        // 返回分析结果
        res.writeHead(200, {
          "Content-Type": "application/json",
          "Access-Control-Allow-Methods": "*",
          "Access-Control-Allow-Origin": "*",
        });
        res.end(JSON.stringify(result));
      } catch (error) {
        console.error("处理上传文件失败:", error);
        res.writeHead(500, {
          "Content-Type": "application/json",
          "Access-Control-Allow-Methods": "*",
          "Access-Control-Allow-Origin": "*",
        });
        res.end(JSON.stringify({ error: "处理上传文件失败" }));
      }
    });
  } catch (error) {
    console.error("文件上传处理失败:", error);
    res.writeHead(500, {
      "Content-Type": "application/json",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Origin": "*",
    });
    res.end(JSON.stringify({ error: "服务器错误" }));
  }
}

// 主后端函数，处理所有请求
async function diseaseAnalysisServer(req, res) {
  // 解析URL
  const parsedUrl = url.parse(req.url);
  const pathname = parsedUrl.pathname;
  
  // 设置CORS头，允许跨域请求
  res.setHeader("Access-Control-Allow-Methods", "*");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  
  // 处理预检请求
  if (req.method === "OPTIONS") {
    res.writeHead(200);
    res.end();
    return;
  }
  
  // 分析API
  if (pathname === "/api/analyze" && req.method === "POST") {
    await handleFileUpload(req, res);
    return;
  }
  
  // 模型状态API
  if (pathname === "/api/model-status" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ 
      loaded: true, 
      status: "模型已加载，可以进行分析"
    }));
    return;
  }
  
  // 文本处理API（保留原有功能）
  if (pathname === "/api/text") {
    res.writeHead(200, { "Content-Type": "application/json;charset=utf-8" });
    
    if (req.method === "GET") {
      if (fs.existsSync("wenZi.txt")) {
        let wenZi = fs.readFileSync("wenZi.txt", "utf8").toString();
        res.end(JSON.stringify({ data: wenZi }));
      } else {
        res.end("{}");
      }
    } else if (req.method === "POST") {
      let xinWenZi = decodeURI(parsedUrl.query || "");
      fs.writeFileSync("wenZi.txt", xinWenZi);
      res.end("{}");
    } else if (req.method === "DELETE") {
      fs.writeFileSync("wenZi.txt", "");
      res.end("{}");
    }
    return;
  }
  
  // 处理静态文件请求（HTML, CSS, JS等）
  if (req.method === "GET") {
    let filePath = path.join(__dirname, pathname);
    
    // 默认访问index.html
    if (pathname === "/") {
      filePath = path.join(__dirname, "index.html");
    }
    
    // 检查文件是否存在
    fs.access(filePath, fs.constants.F_OK, (err) => {
      if (err) {
        // 文件不存在
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("404 Not Found");
        return;
      }
      
      // 读取并提供静态文件
      fs.readFile(filePath, (err, data) => {
        if (err) {
          res.writeHead(500, { "Content-Type": "text/plain" });
          res.end("Server Error");
          return;
        }
        
        // 确定内容类型
        const extname = path.extname(filePath);
        let contentType = "text/html";
        
        switch (extname) {
          case ".js":
            contentType = "text/javascript";
            break;
          case ".css":
            contentType = "text/css";
            break;
          case ".json":
            contentType = "application/json";
            break;
          case ".png":
            contentType = "image/png";
            break;
          case ".jpg":
          case ".jpeg":
            contentType = "image/jpeg";
            break;
        }
        
        // 发送文件
        res.writeHead(200, { "Content-Type": contentType });
        res.end(data);
      });
    });
    return;
  }
  
  // 处理未知请求
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not Found" }));
}
// 下面http对象的createServer函数创建了一个server也就是服务
// 这个服务的所有逻辑交给houDuan函数出处理，请求来的数据和返回的函数会传入houDuan函数
// 监听listen在3000端口上。
// 这个函数比较特殊，它不是一次性的，而是listen监听
// 一直会接收发到3000端口的网络请求。
http.createServer(houDuan).listen(3000);
// 在终端里显示启动成功提示
console.log("服务启动，地址：http://localhost:3000/");
