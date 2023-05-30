# langchain_tutorial

# できること

LangChainを使って、PDFのドキュメントを読み込み、ドキュメントに対する問い合わせができます。

# 前提

* 動作環境
  * Windows10/11 WSL2
* インストール
  * Python3が利用できること
* OpenAI
  * OpenAPI key を利用できること

# インストール方法

以下をインストールして下さい。

```
pip3 install openai
pip3 install chromadb 
pip3 install tiktoken
pip3 install pypdf
```

OpenAPIのAPIキーを環境変数として設定してください。

```
export OPENAI_API_KEY=<APIキー>
```

documentsディレクトリを作成して、その配下に、読み込ませたい PDF ファイルを配置してください。

```
mkdir documents
```

```
mv <PDF files> documents/
```

以下のコマンドでDBを作成します。

```
python3 chat.py new
```

成功すると、こうなります。

例：
```
$ python3 chat.py new
PDFロード：athrill.pdf
PDF ドキュメントの内容を分割する
PDFロード：hakoniwa.pdf
PDF ドキュメントの内容を分割する
PDFロード：newsletter.pdf
PDF ドキュメントの内容を分割する
LangChain における LLM のセットアップ
分割したテキストの情報をベクターストアに格納する
$ ls DB
chroma-collections.parquet  chroma-embeddings.parquet  index
```

# 実行方法

```
python3 chat.py load
```

# デモ

以下のPDFファイルを読みこませました。

* https://www.toppers.jp/newsletter/newsletter-2111.pdf
* https://www.toppers.jp/docs/contest/2022/A01_mori.pdf
* https://www.toppers.jp/docs/contest/2017/athrill.pdf

```
$ python3 chat.py new
$ python3 chat.py load
```

以下、QA内容

> Athrill とは何ですか？

質問：Athrill とは何ですか？
回答：Athrillは、V850 CPUエミュレータであり、TOPPERS/ASP3カーネルを使用して、V850実機レス開発環境を提供するために開発されました。具体的には、CPU命令エミュレーション、デバイスエミュレーション（タイマ、シリアル、CANなど）、デバッグ機能（ソースレベルデバッグ、割り込み発生、プロファイラなど）を備えています。V850は車載系で広く利用されており、TOPPERSソフトウェアを車載向けでも手軽に利用しやすくすることが目的とされています。

> C++版箱庭コア機能とは何ですか？

質問：C++箱庭コア機能とは何ですか？
回答：C++版箱庭コア機能は、箱庭システムの中核となる機能であり、オレンジ部分が箱庭コア部分であり、青色・緑色部分が箱庭コアを利用しやすくするためのコマンドやプロキシなどで構成されます。具体的には、箱庭マスタ制御、ECU間通信を実現するCAN通信デバイス、箱庭CANモニタ・プロキシ、箱庭時間同期用Athrillデバイスなどが含まれます。詳細はGitHubで公開されています。
