from http import HTTPStatus
import dashscope

class QueryTongyi:
    # return format:
    # status: True/False
    # content:
    def __init__(self, api_key):
        dashscope.api_key=api_key

    def query(self, prompt, temperature=0.3, top_p=0.8):
        res = {}
        try:
            response = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_plus,
                # model='qwen1.5-1.8b-chat',
                prompt=prompt,
                temperature=temperature,
                top_p = top_p
            )
            # The response status_code is HTTPStatus.OK indicate success,
            # otherwise indicate request is failed, you can get error code
            # and message from code and message.

            res['status'] = response.status_code == HTTPStatus.OK
            if res['status']:
                res['content'] = response.output.text
            else:
                res['content'] = ""
        except:
            res['status'] = False
            res['content'] = ""
        return res

