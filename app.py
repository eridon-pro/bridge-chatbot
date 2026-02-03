from flask import Flask, render_template, request, jsonify
import os
import base64
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

# デフォルトモデル
DEFAULT_MODEL = 'gemini-3-flash-preview'

# 画像生成対応モデル（モデル名に含まれるキーワード）
IMAGE_GENERATION_MODEL_KEYWORDS = ['image', 'imagen']

@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    """利用可能なモデル一覧を取得"""
    try:
        # APIキーの確認を最初に行う
        if not os.getenv('GEMINI_API_KEY') or not client:
            return jsonify({
                'error': 'Gemini APIキーが設定されていません。.envファイルにGEMINI_API_KEYを設定してください。',
                'status': 'error'
            }), 500
        
        # APIから利用可能なモデルを取得
        models = client.models.list()
        available_models = []
        seen_ids = set()  # 重複チェック用
        
        for model in models:
            model_name = model.name.split('/')[-1] if '/' in model.name else model.name
            # Geminiモデルのみをフィルタリング
            if 'gemini' in model_name.lower() or 'nano' in model_name.lower():
                # 重複チェック
                if model_name in seen_ids:
                    continue
                seen_ids.add(model_name)
                
                # チャット/生成がサポートされているモデルのみを含める
                supported_methods = getattr(model, 'supported_generation_methods', [])
                if not supported_methods or 'generateContent' in supported_methods or 'chat' in str(supported_methods).lower():
                    # オリジナルのモデル名をそのまま使用（読みやすくするためハイフンをスペースに変換）
                    display_name = getattr(model, 'display_name', None) or model_name.replace('-', ' ').replace('_', ' ')
                    
                    available_models.append({
                        'id': model_name,
                        'name': display_name,
                        'description': ''  # 説明文は表示しない
                    })
        
        if not available_models:
            return jsonify({
                'error': '利用可能なモデルが見つかりませんでした。',
                'status': 'error'
            }), 500
        
        # デフォルトモデルを最初に配置
        available_models.sort(key=lambda x: (x['id'] != DEFAULT_MODEL, x['id']))
        return jsonify({'models': available_models, 'status': 'success'})
    
    except Exception as e:
        # APIから取得できない場合はエラーを返す
        error_message = str(e)
        print(f"モデル一覧の取得に失敗しました: {error_message}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'モデル一覧の取得に失敗しました: {error_message}',
            'status': 'error'
        }), 500

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
            available_models = [m.name.split('/')[-1] if '/' in m.name else m.name 
                               for m in client.models.list() 
                               if 'gemini' in (m.name.split('/')[-1] if '/' in m.name else m.name).lower()]
            if model_id not in available_models:
                # 利用可能なモデルが見つからない場合はエラーを返す
                return jsonify({
                    'error': f'モデル "{model_id}" は利用できません。利用可能なモデルを選択してください。'
                }), 400
        except Exception as e:
            # APIから取得できない場合はエラーを返す
            print(f"モデル検証エラー: {e}")
            return jsonify({
                'error': f'モデルの検証に失敗しました: {str(e)}'
            }), 500
        
        # 画像生成モデルかどうかを判定
        is_image_model = any(kw in model_id.lower() for kw in IMAGE_GENERATION_MODEL_KEYWORDS)
        
        # セッションキーにモデルIDを含める（モデル変更時に新しいセッションを作成）
        session_key = f"{session_id}_{model_id}"
        
        # セッションが存在しない、またはモデルが変更された場合は新規作成
        if session_key not in chat_sessions:
            # 画像生成モデルの場合は response_modalities を設定
            if is_image_model:
                chat_config = types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
                chat_sessions[session_key] = {
                    'chat': client.chats.create(model=model_id, config=chat_config),
                    'initialized': False,
                    'is_image_model': True
                }
            else:
                chat_sessions[session_key] = {
                    'chat': client.chats.create(model=model_id),
                    'initialized': False,
                    'is_image_model': False
                }
        
        chat_data = chat_sessions[session_key]
        chat = chat_data['chat']
        is_image_model = chat_data.get('is_image_model', False)
        
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
            
            # テキストと画像を抽出
            ai_response = ''
            images_base64 = []
            
            # response の parts をイテレート（画像生成モデルの場合）
            parts = response.parts if hasattr(response, 'parts') and response.parts else None
            if parts:
                for part in parts:
                    # thought（思考プロセス）の画像はスキップ
                    if getattr(part, 'thought', False):
                        continue
                    if hasattr(part, 'text') and part.text:
                        ai_response += part.text
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                        try:
                            mime = getattr(part.inline_data, 'mime_type', None) or 'image/png'
                            b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                            images_base64.append(f"data:{mime};base64,{b64}")
                        except Exception as img_err:
                            print(f"画像の変換に失敗: {img_err}")
            
            # parts がない場合は従来通り text を取得
            if not ai_response and not images_base64 and hasattr(response, 'text'):
                ai_response = response.text or ''
        except Exception as api_error:
            # より詳細なエラーメッセージを返す
            error_message = str(api_error)
            print(f"API呼び出しエラー: {error_message}")
            print(f"モデル: {model_id}")
            import traceback
            traceback.print_exc()
            
            # google.genai.errorsの特定のエラータイプをチェック
            error_type = type(api_error).__name__
            error_str = error_message.lower()
            
            # 429エラー（クォータ超過）を最優先でチェック
            if ('429' in error_message or 'resource_exhausted' in error_str or
                'RateLimitError' in error_type or 'ClientError' in error_type):
                return jsonify({
                    'error': 'APIのクォータ制限に達しました。しばらく待ってから再試行してください。'
                }), 429
            # ServerError（503など）をチェック
            elif 'ServerError' in error_type or '503' in error_message or 'unavailable' in error_str or 'overloaded' in error_str:
                return jsonify({'error': 'モデルが過負荷です。しばらく待ってから再試行してください。'}), 503
            elif 'NotFoundError' in error_type or '404' in error_message or ('not found' in error_str and 'model' in error_str):
                return jsonify({'error': f'モデルが見つかりません: {model_id}'}), 404
            elif 'AuthenticationError' in error_type or '401' in error_message or 'authentication' in error_str or 'api key' in error_str or 'unauthorized' in error_str:
                return jsonify({'error': 'Gemini APIキーが無効です'}), 401
            elif 'rate limit' in error_str or 'quota' in error_str:
                return jsonify({'error': 'APIのレート制限に達しました。しばらく待ってから再試行してください。'}), 429
            else:
                # その他のエラー
                raise Exception(f"Gemini API呼び出しエラー: {error_message}")
        
        result = {
            'response': ai_response,
            'status': 'success',
            'session_id': session_id,
            'model': model_id
        }
        if images_base64:
            result['images'] = images_base64
        return jsonify(result)
    
    except Exception as e:
        error_str = str(e).lower()
        error_message = str(e)
        
        # エラーの詳細をログに出力
        print(f"エラー詳細: {error_message}")
        print(f"エラータイプ: {type(e).__name__}")
        
        # エラータイプに応じた適切なHTTPステータスコードを返す
        if '503' in error_message or 'unavailable' in error_str or 'overloaded' in error_str:
            return jsonify({'error': 'モデルが過負荷です。しばらく待ってから再試行してください。'}), 503
        elif 'authentication' in error_str or 'api key' in error_str or 'invalid' in error_str or 'unauthorized' in error_str:
            return jsonify({'error': 'Gemini APIキーが無効です'}), 401
        elif '429' in error_message or 'resource_exhausted' in error_str or 'rate limit' in error_str or 'quota' in error_str:
            return jsonify({'error': 'APIのクォータ制限に達しました。しばらく待ってから再試行してください。'}), 429
        elif 'not found' in error_str and 'model' in error_str and '404' in error_message:
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
