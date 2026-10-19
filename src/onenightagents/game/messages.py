"""プレイヤー宛メッセージの生成。情報隔離の唯一の出口(docs/agent-design.md 5)。

渡すのは「公開ログの差分」「本人の秘密(夜の結果)」「いま求める行動と候補」だけ。
他プレイヤーの役職・中央・他者の夜情報・他者の投票は含めない。
"""

from __future__ import annotations

from onenightagents.game.models import (
    CENTER,
    GameState,
    NightRequest,
    NotesRequest,
    Role,
    SpeakRequest,
    Utterance,
    VoteRequest,
)

ROLE_NAMES = {Role.VILLAGER: "村人", Role.WEREWOLF: "人狼", Role.SEER: "占い師", Role.THIEF: "怪盗"}

Request = NightRequest | NotesRequest | SpeakRequest | VoteRequest


def build_message(state: GameState, pid: str, request: Request) -> str:
    parts: list[str] = []
    if _is_first_message(state, pid, request):
        parts.append(_intro(state, pid))
    if isinstance(request, NotesRequest):
        parts.append(_dawn(state, pid))
    diff = _public_diff(state, pid)
    if diff:
        parts.append(diff)
    parts.append(_request_text(state, pid, request))
    return "\n".join(parts)


def _is_first_message(state: GameState, pid: str, request: Request) -> bool:
    # 占い師と怪盗は夜の要求が最初のメッセージ、それ以外は夜明けの通知が最初
    if isinstance(request, NightRequest):
        return True
    return isinstance(request, NotesRequest) and state["initial_roles"][pid] not in (Role.SEER, Role.THIEF)


def _intro(state: GameState, pid: str) -> str:
    config = state["config"]
    names = "、".join(f"{p.id}({p.display_name})" for p in config.players)
    me = next(p for p in config.players if p.id == pid)
    role = ROLE_NAMES[state["initial_roles"][pid]]
    return f"ワンナイト人狼を始めます。参加者: {names}。\nあなたは {me.display_name}(ID: {pid})で、配られた役職は「{role}」です。"


def _dawn(state: GameState, pid: str) -> str:
    night = state["night"]
    role = state["initial_roles"][pid]
    lines = ["夜が明けました。"]
    if role is Role.SEER:
        result = night.seer_result
        if isinstance(result, dict):
            target, seen = next(iter(result.items()))
            lines.append(f"占いの結果: {target} の役職は「{ROLE_NAMES[seen]}」でした。")
        elif isinstance(result, list):
            lines.append("占いの結果: 中央の 2 枚は「" + "」「".join(ROLE_NAMES[r] for r in result) + "」でした。")
    elif role is Role.WEREWOLF:
        partners = [w for w in night.werewolves if w != pid]
        if partners:
            lines.append(f"仲間の人狼: {', '.join(partners)}。")
        else:
            lines.append("あなたは一匹狼です(もう 1 枚の人狼カードは中央にあります)。")
    elif role is Role.THIEF:
        if night.thief_target is None:
            lines.append("交換しませんでした。あなたの役職は怪盗のままです。")
        else:
            lines.append(f"{night.thief_target} と交換しました。あなたの現在の役職は「{ROLE_NAMES[night.thief_new_role]}」です。")
    return "\n".join(lines)


def _public_diff(state: GameState, pid: str) -> str:
    start = state.get("delivered_upto", {}).get(pid, 0)
    new = state.get("utterances", [])[start:]
    if not new:
        return ""
    return "新しい発言:\n" + "\n".join(_format_utterance(u) for u in new)


def _format_utterance(u: Utterance) -> str:
    if u.is_pass:
        return f"- #{u.turn} {u.speaker}: パス"
    meta = []
    if u.addressed_to:
        meta.append(f"宛先: {u.addressed_to}")
    if u.claimed_role:
        meta.append(f"主張: {ROLE_NAMES[u.claimed_role]}")
    suffix = f"({'、'.join(meta)})" if meta else ""
    return f"- #{u.turn} {u.speaker}: 「{u.content}」{suffix}"


def _request_text(state: GameState, pid: str, request: Request) -> str:
    if isinstance(request, NightRequest):
        if request.role is Role.SEER:
            candidates = ", ".join(f"{CENTER}(中央の 2 枚)" if c == CENTER else c for c in request.candidates)
            hint = f"{CENTER}(中央の 2 枚)" if request.hint == CENTER else request.hint
            return (
                f"夜になりました。占う対象を seer_look で 1 つ選んでください。候補: {candidates}。"
                f"候補に優劣はありません。迷う場合は {hint} を選んでください"
            )
        return (
            "夜になりました。カードを交換する相手を thief_swap で選んでください。"
            f"交換しない場合は target を null にしてください。候補: {', '.join(request.candidates)}。"
            f"候補に優劣はありません。迷う場合は {request.hint} を選んでください"
        )
    if isinstance(request, NotesRequest):
        return "これから議論です。/notes.md に推理メモを書いてください。"
    if isinstance(request, SpeakRequest):
        turn = request.turn
        label = "オープニング" if turn.round == 0 else f"自由ラウンド {turn.round}"
        lines = [f"【{label}】"]
        if turn.reply_to_turn is not None:
            asker = state["utterances"][turn.reply_to_turn].speaker
            lines.append(f"あなたは {asker} からの発言 #{turn.reply_to_turn} で問われています。")
        if turn.pass_allowed:
            lines.append("あなたの発言ターンです。speak で発言するか、pass_turn でパスしてください。")
        else:
            lines.append("あなたの発言ターンです。speak で発言してください(このターンはパスできません)。")
        lines.append(f"宛先に選べる相手: {', '.join(request.others)}")
        return "\n".join(lines)
    if isinstance(request, VoteRequest):
        return (
            "議論は終了しました。/notes.md を更新したうえで、処刑したい相手を vote で 1 人選んでください。"
            "対抗のない占い師 CO は、占い結果に矛盾がない限り最も信頼できる情報です。"
            "占われた本人の否定や言い方の強さは、占い師を疑う根拠になりません。"
            f"候補: {', '.join(request.candidates)}"
        )
    raise TypeError(f"unknown request: {type(request)}")
