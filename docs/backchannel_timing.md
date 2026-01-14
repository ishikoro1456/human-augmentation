# 相槌の位置と「返さない」を正解に入れる案

このメモは、相槌の不自然さを減らすために、どこで返すかと、返さない選択をどう入れるかを整理します。実装に落とす前の、判断の土台にするのが目的です。

## 1. 「適当な相槌位置」とは何か

会話研究では、話し手の発話の中に、次の人が入ってもよい場所があると考えます。これを TRP（Transition Relevance Place）と呼びます。大雑把に言うと、文や節が終わりそうで、意味も一区切りしたと聞き手が見える場所です。Sacks らのターン交代の枠組みでは、聞き手は話し手の発話を聞きながら、いつ一区切りになるかを予測して、そこで交代や短い反応を入れると整理されます。

相槌は、ターンを奪う発話ではなく、話し手を支える短い反応です。ただし、入れる場所が悪いと割り込みに聞こえます。なので、相槌にも「入りやすい位置」があります。

日本語の相槌（あいづち）については、発話の途中に入るものと、発話の最後に入るものが区別されるという説明があります。発話途中の相槌は、話し手に続けてよいと伝える用途が中心で、同意や共感とは別、と整理されています。また、話し手側にも相槌を促す形があり、相槌は話し手と聞き手の組で起きる、と述べられています。

さらに、相槌が入りやすい手がかりとして、文法と音の両方が関係することが示されています。日本語の対話データを使った分析では、品詞などの文法情報と、長さやピッチ（高さ）の動きなどの音の情報が、ターン交代や相槌が起きる場所と関係すると報告されています。別の研究では、英語と日本語で、発話の後半に低いピッチの区間があることが、相槌を引き出す手がかりになり得ると述べられています。

まとめると、適当な相槌位置は、次のように言えます。

- 話し手の発話が一区切りしそうで、聞き手が入りやすい場所（TRP）
- 話し手側の「相槌を入れてよい」合図が出やすい場所
- 文法の切れ目と、音の切れ目が重なることが多い場所

この3つは同じ意味ではありませんが、実装の目標としては同じ方向を向きます。

## 2. 「返さない」を正解に入れるとは、どういうことか

ここで言う「返さない」は、失敗の言い訳ではなく、会話として自然な行動の一つとして扱う、という意味です。

実装としては、二つの入れ方があります。

### 2.1 プロンプトに入れる（モデルの判断に期待する）

モデルに、返事を出す以外に、何もしない選択があると明示します。

- 返すか返さないかを最初に判断する
- 迷ったら返さない
- 返すときも、短く、内容に踏み込みすぎない

この入れ方は手軽ですが、モデルが毎回きちんと守るかは、実験で確かめる必要があります。

### 2.2 出力の形式として入れる（行動の選択肢にする）

モデルの出力に、音声を返す id だけでなく、NONE のような明示的な「返さない」を含めます。こうすると、後段が安定します。返さない場合は、再生しないだけで済みます。

このやり方は、プロンプトよりも強く「返さない」を実装に組み込めます。

## 3. 自己点検（メタ認知）を入れるなら何を点検するか

自己点検は、生成した相槌が不自然になりやすいところを止める役です。点検の軸は、できるだけ少なくして、失敗しにくくするのがよいです。

私は、まず次の3つが効くと思います。

1. それは相槌か
   - 内容への回答になっていないか
   - 説明や提案になっていないか
   - 評価や断定が強すぎないか

2. タイミングは合っているか
   - 相槌位置に入るのが自然か
   - 途中で遮っていないか

3. その場の意図と合うか
   - IMU の動きが示す意図と矛盾していないか
   - 直近の文字起こしの流れと矛盾していないか

実装の形は二つに分けられます。

- 同じモデルで二回目に点検する（提案を作る役と止める役を分ける）
- 役割を分けた別のエージェントが、相槌を拒否したり、NONE を選ばせたりする

合議制は、ここに使うと意味が出やすいです。返事を上手に作る役より、無理な返事を止める役の方が、AIっぽさを下げるのに効く場面があります。

## 4. このプロジェクトへの落とし込み（いまの実装とつなぐ）

いまの実装は、transcribe.txt を TTS で読み上げ、IMU の反応が来たときに「今まで読み上げた文字起こし」と「IMU」をモデルへ渡します。この方式だと、相槌位置は TTS の区切りに寄りやすくなります。

次の実験は、ここから始めるのが良いと思います。

- NONE を許したとき、相槌の頻度がどれくらい自然に見えるか
- NONE を増やすと、話し手が話しやすくなるか
- IMU を隠した場合と比べて、選び方が変わるか

## 5. 既存事例からの学び（うまい相槌は何を見ているか）

ここをはっきりさせたいです。相槌の「タイミング」は、文字の意味だけで決まるものではありません。研究では、相手の声の特徴が手がかりになる、と繰り返し書かれています。たとえば英語と日本語の両方で、発話の後半に低いピッチの区間があると、それが合図になり得る、という報告があります。また、単純な音の手がかりだけで相槌を返す仕組みでも、実際に会話相手から「自然」と感じられる可能性がある、というシステム報告もあります。

日本語は相槌が多い、と言われがちですが、これは印象だけではなく、対話データを見た分析があります。日本語と米国英語を比べた研究では、日本語の方が聞き手の短い反応が多い、と結論づけられています。別の研究でも、英語・日本語・中国語で、短い聞き手の反応の種類や頻度、どこに出るかが違う、と整理されています。

最近の流れとしては、もっとはっきりしています。相槌を「文の意味」だけで当てようとするより、音声をフレーム単位で見て、いつ相槌が出やすいかを連続的に予測する研究が出ています。さらに、音声のモデルと言語モデルを組み合わせると良い、という報告もあります。つまり、タイミングだけは、音声の手がかり抜きで当てるのが難しい、という方向に研究が集まっています。

一方で、文字や語の情報が役に立たないわけではありません。音の特徴だけで予測する仕組みに、単語の情報を足すと性能が上がったという報告があります。日本語でも、文法の情報と音の情報の両方が、相槌が出る場所と関係すると報告されています。なので、疑問詞や疑問文の形、終助詞、感情語彙のような手がかりは、タイミングの主役ではないとしても、相槌の種類を選ぶ材料としては使えそうです。

## 6. 「重要な情報を漏らさず渡す」ための設計メモ

私の意見ですが、今のつまずきは「コンテクスト不足」だけでは説明しにくいです。タイミングに必要な情報が、そもそも入力に無いか、入力の形が合っていない可能性があります。

そこで、入力を次の3つに分けて考えるのが良いと思います。

1. タイミング用の情報
   - 相手の声の切れ目や沈黙、強調など（いまの試作だと TTS の区切りでもよい）
   - 直近の動きの強さ（IMU）と、その変化

2. 内容用の情報
   - 直近の文と、その前後の流れ（短い要約でもよい）
   - いま話している話題、何を言い終えたか、次に言いそうか

3. 方針（出しすぎないための情報）
   - 直近に相槌を出したか
   - いまは音声が重なって聞き取りにくいか
   - 迷ったら NONE に倒す、という安全側の方針

## 7. 次に進むための「調査→設計→実験」の順番

ここから先は、実装を増やす前に、順番を決めた方が良いと思います。

- 調査: 既存研究が「タイミング」を何で当てているかを押さえる（音声の手がかり中心）
- 設計: タイミングと内容を分け、どの入力がどちらに必要かを決める
- 実験: ログを取り、どこでズレたかを後から説明できるようにする

この順番にすると、試行錯誤が「反省できる試行錯誤」になります。

## 8. 今後のタスク順（訓練なしで進める前提）

ここは「汎用モデルで工夫する」ための、実装の順番案です。ポイントは、モデルに丸投げしないで、判断しやすい材料と場面を作ることです。

1. ログの形を固定する
   - 何が起きたかを、後から説明できないと改善できません。相槌はタイミングが大事なので、特に必要です。
   - 取るもの: 今の文、直近の文脈の要約、IMU要約、相槌を出したか、出したなら音が重なったか、理由。

2. 相槌の候補点をしぼる（安全な瞬間だけ判断する）
   - 話を遮るのが一番の害なので、まずそこを減らします。
   - 例: TTS の1チャンクが終わった直後だけ判断する。音声が鳴っているときは、原則 NONE。

3. 疑問と感情の手がかりを「特徴」として作って渡す
   - これはルールで決めるのではなく、汎用モデルが判断しやすい材料を増やすためです。
   - 例: 疑問文らしさ（疑問詞、疑問の終わり方）、感情らしさ（驚き、喜び、困り、怒りなど）を短いタグで渡す。
   - 注意: これだけで相槌のタイミングを当てるのは難しいので、あくまで「内容と種類」を決める材料として使います。

4. 最初に測定して個人差を吸収する（パーソナライズ）
   - 同じ動きでも人によって大きさや頻度が違います。最初に基準を取って「その人にとっての普通」を作ります。
   - 例: 30秒ほど静止、次に1〜2分ほど普段どおりに聞く。そこから平均とばらつきを作り、現在値を相対値で渡す。

5. 自己点検の段を足して、怪しい相槌を止める
   - うまい相槌を作るより、変な相槌を止める方が効果が出やすいです。
   - 例: 1回目で候補を出し、2回目で「遮るか」「内容に踏み込みすぎか」を見て、だめなら NONE。

## 参考文献（リンク）

- Ruede et al. (2017) Enhancing Backchannel Prediction Using Word Embeddings. Interspeech. https://www.isca-archive.org/interspeech_2017/ruede17_interspeech.html
- Ortega et al. (2023) Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion: An Initial Investigation. arXiv. https://arxiv.org/abs/2304.04478

- Sacks, Schegloff, Jefferson (1974) A simplest systematics for the organization of turn-taking for conversation. Language. https://www.conversationanalysis.org/schegloff-media-archive/simplest-systematics-for-turn-taking-language-1974/
- Koiso et al. (1998) An Analysis of Turn-Taking and Backchannels Based on Prosodic and Syntactic Features in Japanese Map Task Dialogs. Language and Speech. https://pubmed.ncbi.nlm.nih.gov/10746360/
- Ward & Tsukahara (2000) Prosodic Features which Cue Back-channel Responses in English and Japanese. Journal of Pragmatics. https://www.cs.utep.edu/nigel/abstracts/jprag00.html
- Ward & Tsukahara (1999) A Responsive Dialog System. in Machine Conversations. https://www.cs.utep.edu/nigel/abstracts/bellagio
- White (1989) Backchannels across cultures: A study of Americans and Japanese. Language in Society. https://doi.org/10.1017/S0047404500013270
- Blomsma et al. (2024) Backchannel behavior is idiosyncratic. Language and Cognition. https://doi.org/10.1017/langcog.2024.1
- Cook (2000) The particle ne as a turn-management device in Japanese conversation. Journal of Pragmatics. https://doi.org/10.1016/S0378-2166(99)00087-9
- Ide & Kawahara (2022) Building a Dialogue Corpus Annotated with Expressed and Experienced Emotions. arXiv. https://arxiv.org/abs/2205.11867
- Columbia Games Corpus: Question Function Guidelines (Rhetorical Question/Backchannel). https://www.cs.columbia.edu/speech/games-corpus/guidelines-question-function.php
- Huang et al. (2025) CMIS-Net: A Cascaded Multi-Scale Individual Standardization Network for Backchannel Agreement Estimation. arXiv. https://arxiv.org/abs/2510.17855
- Miyata (2007) The acquisition of Japanese backchanneling behavior: ... Journal of Pragmatics. https://www.sciencedirect.com/science/article/pii/S0378216607000525
- Maynard (1990) Conversation management in contrast: Listener response in Japanese and American English. Journal of Pragmatics. https://doi.org/10.1016/0378-2166(90)90097-W
- Clancy et al. (1996) The conversational use of reactive tokens in English, Japanese, and Mandarin. Journal of Pragmatics. https://doi.org/10.1016/0378-2166(95)00036-4
- Inoue et al. (2024) Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection. arXiv. https://arxiv.org/abs/2410.15929
- Wang et al. (2024) Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion. arXiv. https://arxiv.org/abs/2401.14717
- Inoue et al. (2025) Multilingual and Continuous Backchannel Prediction: A Cross-lingual Study. arXiv. https://arxiv.org/abs/2512.14085
