def generation(theme, count, examples):
    text = ("日本語の大喜利のお題を作成してください。一読して理解でき、複数の異なる回答が可能な問いにしてください。"
            "差別、露骨な性表現、残虐な暴力、実在の個人への攻撃を避けてください。"
            "回答、解説、番号は含めず、お題のみを JSON の items 配列に入れてください。"
            f"テーマ: {theme or '自由'}。件数: {count}。")
    if examples:
        text += "参考例の文言や固有名詞は再利用しないでください。参考例: " + " / ".join(examples)
    return [{"role": "user", "content": text}]


def answer(prompt):
    return [{"role": "user", "content": f"次の大喜利のお題に対し、異なる回答を3つ考え、JSONのitems配列で返してください。\nお題: {prompt}"}]
