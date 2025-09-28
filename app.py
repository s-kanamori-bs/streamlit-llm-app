import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

sports_template = """
あなたはダイエットに関する運動の専門家です。
効率よくダイエットをする為の運動をアドバイスします。
質問：{input}
"""

eat_template = """
あなたはダイエットに関する食事の専門家です。
ダイエット中に無理なく体重を落とす為の食事をアドバイスします。
質問：{input}
"""

life_template = """
あなたはダイエットに関する生活習慣の専門家です。
ダイエットするに辺り、日常生活で気を付けることをアドバイスします。
質問：{input}
"""

prompt_infos = [
    {
        "name": "sports",
        "description": "ダイエットに関する運動の専門家です",
        "prompt_template": sports_template
    },
    {
        "name": "eat",
        "description": "ダイエットに関する食事の専門家です",
        "prompt_template": eat_template
    },
    {
        "name": "life",
        "description": "ダイエットに関する生活習慣の専門家です",
        "prompt_template": life_template
    },
]

def get_llm_response(expert_type, user_input):
    """
    選択された専門家とユーザー入力を基にLLMからの回答を取得する関数
    
    Args:
        expert_type (str): 選択された専門家のタイプ
        user_input (str): ユーザーの入力テキスト
    
    Returns:
        str: LLMからの回答
    """
    import os
    
    # OpenAI APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。"
    
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
        
        # 対応するテンプレートを取得
        template = None
        for info in prompt_infos:
            if info["name"] == expert_type:
                template = info["prompt_template"]
                break
        
        if template:
            prompt = PromptTemplate(template=template, input_variables=["input"])
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(input=user_input)
            return response
        else:
            return "申し訳ございませんが、該当する専門家が見つかりませんでした。"
            
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}\n\nOpenAI APIキーが正しく設定されているか確認してください。"

# Streamlitアプリケーション
def main():
    st.title("🏃‍♂️ ダイエット専門家相談アプリ")
    st.markdown("専門家を選択して、あなたの質問や悩みを相談してください！")
    
    # ラジオボタンで専門家を選択
    expert_options = [
        ("sports", "🏋️‍♀️ 運動の専門家 - 効率よくダイエットをする為の運動をアドバイス"),
        ("eat", "🥗 食事の専門家 - ダイエット中に無理なく体重を落とす為の食事をアドバイス"),
        ("life", "🛏️ 生活習慣の専門家 - ダイエットする際の日常生活での注意点をアドバイス")
    ]
    
    selected_expert = st.radio(
        "相談したい専門家を選択してください：",
        options=[option[0] for option in expert_options],
        format_func=lambda x: next(option[1] for option in expert_options if option[0] == x)
    )
    
    # テキスト入力欄
    user_input = st.text_area(
        "質問や相談内容を入力してください：",
        placeholder="例：効果的な運動方法を教えてください、健康的なダイエット食事メニューを知りたいです、など",
        height=100
    )
    
    # 送信ボタン
    if st.button("相談する", type="primary"):
        if user_input.strip():
            with st.spinner("専門家が回答を考えています..."):
                response = get_llm_response(selected_expert, user_input)
            
            st.markdown("### 📝 専門家からの回答")
            st.markdown(response)
        else:
            st.warning("質問内容を入力してください。")

if __name__ == "__main__":
    main()