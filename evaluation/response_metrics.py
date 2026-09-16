import re


def basic_response_checks(df):
    replies = df["reply"].fillna("").astype(str)
    nonempty = float((replies.str.strip() != "").mean())
    has_link_placeholder = float(replies.str.contains(r"URL|http", case=False, regex=True).mean())
    generic_fallback = float(replies.str.contains(r"Thanks for reaching out", case=False, regex=True).mean())
    return {
        "nonempty_rate": nonempty,
        "contains_reference_or_link_rate": has_link_placeholder,
        "generic_fallback_rate": generic_fallback,
    }
