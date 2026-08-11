# 3D Inference API

This is the API reference for the deployable queue snapshot in this directory.

本文档描述通过统一持久化队列调用 TRELLIS 和 Hunyuan3D 的方式。

## 在线文档

服务启动后可以直接访问：

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI JSON: `http://localhost:8080/openapi.json`

局域网客户端请将 `localhost` 替换为运行服务的 Mac IP。客户端只应访问
`8080`；`8081` 和 `8082` 是本机内部后端。

## 基本流程

1. 将输入图片编码为 Base64。
2. `POST /submit`，保存响应中的 `uid`。
3. 轮询 `GET /status/{uid}`。
4. 当状态为 `completed` 时，请求 `GET /download/{uid}` 获取 GLB 文件。

结果默认保留 24 小时，应在完成后及时下载。

## 认证

未配置 `INFERENCE_API_KEY` 时不需要认证。配置后，除 `/health` 和
`/ready` 外的请求都必须包含：

```http
X-API-Key: YOUR_API_KEY
```

以下示例用环境变量表示可选的认证 Header：

```bash
AUTH_HEADER="X-API-Key: $INFERENCE_API_KEY"
```

## Endpoint

| Method | Path | 用途 |
| --- | --- | --- |
| `POST` | `/submit` | 提交 TRELLIS 或 Hunyuan3D 任务 |
| `GET` | `/status/{uid}` | 查询任务状态和队列位置 |
| `GET` | `/download/{uid}` | 下载完成的 GLB 文件 |
| `POST` | `/cancel/{uid}` | 取消尚未开始的任务 |
| `GET` | `/queue` | 查看队列和各状态任务数量 |
| `GET` | `/health` | 检查队列进程和 SQLite |
| `GET` | `/ready` | 检查 SQLite 和两个模型后端 |
| `POST` | `/send` | 旧版兼容接口，不建议新客户端使用 |

## 提交任务

### 公共字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | string | 是 | - | `trellis` 或 `hunyuan3d` |
| `image` | string | 是 | - | 原始图片内容的 Base64，最大 25 MiB |
| `seed` | integer | 否 | `42` | `0` 到 `4294967295` |

### TRELLIS 字段

| 字段 | 类型 | 默认值 | 可选值/范围 |
| --- | --- | --- | --- |
| `pipeline_type` | string | `512` | `512`, `1024`, `1024_cascade` |
| `texture_size` | integer | `1024` | `512`, `1024`, `2048` |
| `no_texture` | boolean | `false` | `true` 时只生成几何 |
| `steps` | integer/null | `null` | `1` 到 `100`；空值使用模型默认值 |

```bash
IMAGE_BASE64=$(base64 < input.png | tr -d '\n')

curl http://localhost:8080/submit \
  -H 'Content-Type: application/json' \
  -H "$AUTH_HEADER" \
  -d "{
    \"model\": \"trellis\",
    \"image\": \"$IMAGE_BASE64\",
    \"pipeline_type\": \"512\",
    \"texture_size\": 1024,
    \"no_texture\": false,
    \"seed\": 42
  }"
```

### Hunyuan3D 字段

| 字段 | 类型 | 默认值 | 可选值/范围 |
| --- | --- | --- | --- |
| `texture` | boolean | `true` | 是否生成 PBR 纹理 |
| `num_inference_steps` | integer | `50` | `1` 到 `200` |
| `guidance_scale` | number | `7.5` | `0.1` 到 `20.0` |
| `octree_resolution` | integer | `256` | `64` 到 `512` |

```bash
curl http://localhost:8080/submit \
  -H 'Content-Type: application/json' \
  -H "$AUTH_HEADER" \
  -d "{
    \"model\": \"hunyuan3d\",
    \"image\": \"$IMAGE_BASE64\",
    \"texture\": true,
    \"num_inference_steps\": 50,
    \"guidance_scale\": 7.5,
    \"octree_resolution\": 256,
    \"seed\": 42
  }"
```

成功响应：

```json
{
  "uid": "005ca016-1105-4061-be06-c7fef731381d",
  "model": "trellis"
}
```

## 查询状态

```bash
curl -H "$AUTH_HEADER" \
  http://localhost:8080/status/005ca016-1105-4061-be06-c7fef731381d
```

可能的状态：

| 状态 | 含义 |
| --- | --- |
| `queued` | 正在持久化队列中等待；响应包含 `position` |
| `dispatching` | 正在向模型后端派发 |
| `processing` | 后端正在加载模型或执行推理 |
| `completed` | 已完成；响应包含 `download_url` |
| `error` | 失败；响应包含 `message` |
| `cancelled` | 排队期间被取消 |
| `expired` | 任务记录仍存在，但结果文件已过保留期 |

完成响应：

```json
{
  "status": "completed",
  "model": "trellis",
  "attempts": 1,
  "download_url": "/download/005ca016-1105-4061-be06-c7fef731381d"
}
```

## 下载结果

```bash
curl -L -H "$AUTH_HEADER" \
  http://localhost:8080/download/005ca016-1105-4061-be06-c7fef731381d \
  -o result.glb
```

成功时返回 `Content-Type: model/gltf-binary`。任务仍在运行时返回 `202`，结果
已过期时返回 `410`。

## 完整 Python 示例

```python
import base64
import json
import time
import urllib.request

BASE_URL = "http://localhost:8080"
API_KEY = None


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=120)


with open("input.png", "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode("ascii")

with request("/submit", {
    "model": "trellis",
    "image": image,
    "pipeline_type": "512",
    "texture_size": 1024,
}) as response:
    uid = json.loads(response.read())["uid"]

while True:
    with request(f"/status/{uid}") as response:
        status = json.loads(response.read())
    if status["status"] == "completed":
        break
    if status["status"] in ("error", "cancelled", "expired"):
        raise RuntimeError(status)
    time.sleep(5)

with request(f"/download/{uid}") as response, open("result.glb", "wb") as output:
    while chunk := response.read(1024 * 1024):
        output.write(chunk)
```

## 错误码

| HTTP 状态码 | 含义 |
| --- | --- |
| `400` | Base64 图片无效或任务不能取消 |
| `401` | API Key 缺失或错误 |
| `404` | UID 不存在或任务失败 |
| `410` | 结果文件已过期 |
| `413` | 解码后的图片超过大小限制 |
| `422` | 参数类型、枚举或范围不合法 |
| `503` | SQLite 或某个模型后端未就绪 |

## Hunyuan3D-Part 网格分割

Part 接口直接接收 GLB/OBJ 二进制，不使用 JSON Base64。`8080` 将输入持久化后转交给仅监听本机 `8083` 的 MLX worker。

```bash
curl -X POST \
  'http://localhost:8080/submit/part?filename=chair.glb&mode=segment' \
  -H 'Content-Type: model/gltf-binary' \
  -H "$AUTH_HEADER" \
  --data-binary @chair.glb
```

`mode=segment` 返回 P3-SAM 面实例分割；`mode=generate_parts` 运行完整 X-Part。可用 query 参数覆盖：

| 参数 | 默认值 | 范围/说明 |
| --- | ---: | --- |
| `points` | 100000 | 1000–200000 |
| `prompts` | 400 | 1–1000 |
| `prompt_batch_size` | 8 | 1–128 |
| `surface_points` | 81920 | X-Part 条件采样点数 |
| `steps` | 50 | X-Part 流匹配步数 |
| `resolution` | 128 | 128、256 或 512 |
| `sdf_chunk_size` | 100000 | ShapeVAE 解码 chunk |
| `seed` | 42 | 随机种子 |
| `official_fps_start` | true | 使用官方 seeded random FPS 起点 |
| `clean_mesh` | true | 执行官方 merge/process mesh cleaning |
| `connectivity` | true | 启用官方源面投票与连通域修复 |
| `postprocess` | true | 启用官方累计面积小件合并 |
| `postprocess_threshold` | 0.95 | 官方累计面积阈值，范围 0–1 |

轮询方式与图生 3D 相同。完成后 `/download/{uid}` 返回主 GLB；Part 任务还会在 `/bundle/{uid}` 返回完整 ZIP，包括标签数组、包围盒、运行统计和 GLB。
