from flask import Flask, render_template, request, jsonify
import os
import argparse
from dotenv import load_dotenv
from google import genai

# 環境変数を読み込む
load_dotenv()

app = Flask(__name__)

# Gemini APIクライアントを初期化（APIキーがある場合のみ）
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key) if api_key else None

# チャットセッションを保存（本番環境ではデータベースを使用することを推奨）
# セッションごとにチャット履歴を管理
chat_sessions = {}

# システムプロンプト
SYSTEM_INSTRUCTION = 'あなたは親切で役立つアシスタントです。日本語で応答してください。'

# 利用可能なGeminiモデルのリスト（フォールバック用）
AVAILABLE_MODELS = [
    {'id': 'gemini-2.5-flash', 'name': 'Gemini 2.5', 'description': ''},
    {'id': 'gemini-2.5-pro', 'name': 'Gemini 2.5', 'description': ''},
    {'id': 'gemini-2.5-flash-lite', 'name': 'Gemini 2.5', 'description': ''},
    {'id': 'gemini-3-pro-preview', 'name': 'Gemini 3', 'description': ''},
    {'id': 'gemini-3-flash-preview', 'name': 'Gemini 3', 'description': ''},
]

# デフォルトモデル
DEFAULT_MODEL = 'gemini-2.5-flash'

@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    """利用可能なモデル一覧を取得"""
    try:
        # APIから利用可能なモデルを取得を試みる
        try:
            if client and os.getenv('GEMINI_API_KEY'):
                models = client.models.list()
                available_models = []
                for model in models:
                    model_name = model.name.split('/')[-1] if '/' in model.name else model.name
                    # Geminiモデルのみをフィルタリング
                    if 'gemini' in model_name.lower():
                        # チャット/生成がサポートされているモデルのみを含める
                        supported_methods = getattr(model, 'supported_generation_methods', [])
                        if not supported_methods or 'generateContent' in supported_methods or 'chat' in str(supported_methods).lower():
                            # モデル名からハイフン以降を削除
                            display_name = getattr(model, 'display_name', None) or model_name.replace('-', ' ').title()
                            if '-' in display_name:
                                display_name = display_name.split('-')[0].strip()
                            
                            available_models.append({
                                'id': model_name,
                                'name': display_name,
                                'description': ''  # 説明文は表示しない
                            })
                
                if available_models:
                    # デフォルトモデルを最初に配置
                    available_models.sort(key=lambda x: (x['id'] != DEFAULT_MODEL, x['id']))
                    return jsonify({'models': available_models, 'status': 'success'})
        except Exception as e:
            # APIから取得できない場合は、定義済みのリストを返す
            print(f"モデル一覧の取得に失敗しました: {e}")
            import traceback
            traceback.print_exc()
        
        # フォールバック: 定義済みのモデルリストを返す
        return jsonify({'models': AVAILABLE_MODELS, 'status': 'success'})
    
    except Exception as e:
        return jsonify({'error': f'モデル一覧の取得に失敗しました: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """チャットメッセージを処理"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')  # セッションID（デフォルトは'default'）
        model_id = data.get('model', DEFAULT_MODEL)  # 選択されたモデル
        
        if not user_message:
            return jsonify({'error': 'メッセージが空です'}), 400
        
        # Gemini APIキーの確認
        if not os.getenv('GEMINI_API_KEY') or not client:
            return jsonify({
                'error': 'Gemini APIキーが設定されていません。.envファイルにGEMINI_API_KEYを設定してください。'
            }), 500
        
        # モデルの検証（実際に利用可能なモデルを確認）
        try:
            # APIから実際に利用可能なモデルを取得
            if client:
                available_models = [m.name.split('/')[-1] if '/' in m.name else m.name 
                                   for m in client.models.list() 
                                   if 'gemini' in (m.name.split('/')[-1] if '/' in m.name else m.name).lower()]
                if model_id not in available_models:
                    # 利用可能なモデルが見つからない場合はデフォルトを使用
                    if DEFAULT_MODEL in available_models:
                        model_id = DEFAULT_MODEL
                    elif available_models:
                        model_id = available_models[0]
                    else:
                        # APIから取得できない場合は、定義済みリストから検証
                        valid_model_ids = [m['id'] for m in AVAILABLE_MODELS]
                        if model_id not in valid_model_ids:
                            model_id = DEFAULT_MODEL
        except Exception as e:
            # APIから取得できない場合は、定義済みリストから検証
            print(f"モデル検証エラー: {e}")
            valid_model_ids = [m['id'] for m in AVAILABLE_MODELS]
            if model_id not in valid_model_ids:
                model_id = DEFAULT_MODEL
        
        # セッションキーにモデルIDを含める（モデル変更時に新しいセッションを作成）
        session_key = f"{session_id}_{model_id}"
        
        # セッションが存在しない、またはモデルが変更された場合は新規作成
        if session_key not in chat_sessions:
            # チャットセッションを作成
            chat_sessions[session_key] = {
                'chat': client.chats.create(model=model_id),
                'initialized': False
            }
        
        chat_data = chat_sessions[session_key]
        chat = chat_data['chat']
        
        # 初回メッセージの場合はシステムプロンプトを最初に送信（応答は無視）
        if not chat_data['initialized']:
            try:
                chat.send_message(SYSTEM_INSTRUCTION)
                chat_data['initialized'] = True
            except Exception as init_error:
                # システムプロンプトの送信に失敗しても続行
                print(f"システムプロンプトの送信に失敗しましたが、続行します: {init_error}")
                chat_data['initialized'] = True
        
        # チャットセッションにユーザーメッセージを送信
        try:
            response = chat.send_message(user_message)
            # AIの応答を取得
            ai_response = response.text
        except Exception as api_error:
            # より詳細なエラーメッセージを返す
            error_message = str(api_error)
            print(f"API呼び出しエラー: {error_message}")
            print(f"モデル: {model_id}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Gemini API呼び出しエラー: {error_message}")
        
        return jsonify({
            'response': ai_response,
            'status': 'success',
            'session_id': session_id,
            'model': model_id
        })
    
    except Exception as e:
        error_str = str(e).lower()
        error_message = str(e)
        
        # エラーの詳細をログに出力
        print(f"エラー詳細: {error_message}")
        print(f"エラータイプ: {type(e).__name__}")
        
        # エラータイプに応じた適切なHTTPステータスコードを返す
        if 'authentication' in error_str or 'api key' in error_str or 'invalid' in error_str or 'unauthorized' in error_str:
            return jsonify({'error': 'Gemini APIキーが無効です'}), 401
        elif 'rate limit' in error_str or 'quota' in error_str:
            return jsonify({'error': 'APIのレート制限に達しました。しばらく待ってから再試行してください。'}), 429
        elif 'not found' in error_str or 'model' in error_str:
            return jsonify({'error': f'モデルが見つかりません: {error_message}'}), 404
        else:
            # その他のエラーは詳細を返す（デバッグ用）
            return jsonify({'error': f'エラーが発生しました: {error_message}'}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    """チャット履歴をクリア"""
    global chat_sessions
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    
    # 指定されたセッションIDで始まるすべてのセッションを削除（モデルごとのセッションも含む）
    keys_to_delete = [key for key in chat_sessions.keys() if key.startswith(f"{session_id}_")]
    for key in keys_to_delete:
        # セッションが辞書形式（chatとinitializedを含む）の場合はそのまま削除
        del chat_sessions[key]
    
    return jsonify({'status': 'success', 'message': 'チャット履歴をクリアしました'})

if __name__ == '__main__':
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='AIチャットボットアプリケーション')
    parser.add_argument('--debug', action='store_true', help='デバッグモードを有効にする')
    parser.add_argument('--host', default='0.0.0.0', help='ホストアドレス（デフォルト: 0.0.0.0）')
    parser.add_argument('--port', type=int, default=5000, help='ポート番号（デフォルト: 5000）')
    
    args = parser.parse_args()
    
    # デバッグモードは引数で指定された場合のみ有効（デフォルトはFalse）
    app.run(debug=args.debug, host=args.host, port=args.port)
