import sys
sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')
from llm_tool.i18n import get_msg, MESSAGES


def test_get_msg_italian_welcome():
    """
    Messaggio welcome in italiano.
    Verifica che il messaggio di benvenuto sia restituito correttamente in italiano.
    """
    print("\n=== test_get_msg_italian_welcome ===")
    print("Testing: Messaggio welcome in italiano")
    
    result = get_msg("it", "welcome")
    expected = MESSAGES["it"]["welcome"]
    
    print(f"Input: lang_code='it', key='welcome'")
    print(f"Expected: {repr(expected[:50])}...")
    print(f"Actual:   {repr(result[:50])}...")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_english_welcome():
    """
    Messaggio welcome in inglese.
    Verifica che il messaggio di benvenuto sia restituito correttamente in inglese.
    """
    print("\n=== test_get_msg_english_welcome ===")
    print("Testing: Messaggio welcome in inglese")
    
    result = get_msg("en", "welcome")
    expected = MESSAGES["en"]["welcome"]
    
    print(f"Input: lang_code='en', key='welcome'")
    print(f"Expected: {repr(expected[:50])}...")
    print(f"Actual:   {repr(result[:50])}...")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_italian_ask_zone():
    """
    Messaggio ask_zone in italiano.
    Verifica che il messaggio per richiedere la zona sia corretto in italiano.
    """
    print("\n=== test_get_msg_italian_ask_zone ===")
    print("Testing: Messaggio ask_zone in italiano")
    
    result = get_msg("it", "ask_zone")
    expected = MESSAGES["it"]["ask_zone"]
    
    print(f"Input: lang_code='it', key='ask_zone'")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_english_ask_zone():
    """
    Messaggio ask_zone in inglese.
    Verifica che il messaggio per richiedere la zona sia corretto in inglese.
    """
    print("\n=== test_get_msg_english_ask_zone ===")
    print("Testing: Messaggio ask_zone in inglese")
    
    result = get_msg("en", "ask_zone")
    expected = MESSAGES["en"]["ask_zone"]
    
    print(f"Input: lang_code='en', key='ask_zone'")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_italian_oos_fallback():
    """
    Messaggio oos_fallback in italiano.
    Verifica che il messaggio di fallback per richieste fuori tema sia corretto in italiano.
    """
    print("\n=== test_get_msg_italian_oos_fallback ===")
    print("Testing: Messaggio oos_fallback in italiano")
    
    result = get_msg("it", "oos_fallback")
    expected = MESSAGES["it"]["oos_fallback"]
    
    print(f"Input: lang_code='it', key='oos_fallback'")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_english_oos_fallback():
    """
    Messaggio oos_fallback in inglese.
    Verifica che il messaggio di fallback per richieste fuori tema sia corretto in inglese.
    """
    print("\n=== test_get_msg_english_oos_fallback ===")
    print("Testing: Messaggio oos_fallback in inglese")
    
    result = get_msg("en", "oos_fallback")
    expected = MESSAGES["en"]["oos_fallback"]
    
    print(f"Input: lang_code='en', key='oos_fallback'")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_italian_param_error():
    """
    Param error con placeholder.
    Verifica che il param_error italiano sostituisca correttamente il placeholder {0}.
    """
    print("\n=== test_get_msg_italian_param_error ===")
    print("Testing: Param error con placeholder")
    
    result = get_msg("it", "param_error", "formato non valido")
    expected = MESSAGES["it"]["param_error"].format("formato non valido")
    
    print(f"Input: lang_code='it', key='param_error', args=('formato non valido',)")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_english_param_error():
    """
    Param error con placeholder.
    Verifica che il param_error inglese sostituisca correttamente il placeholder {0}.
    """
    print("\n=== test_get_msg_english_param_error ===")
    print("Testing: Param error con placeholder")
    
    result = get_msg("en", "param_error", "invalid format")
    expected = MESSAGES["en"]["param_error"].format("invalid format")
    
    print(f"Input: lang_code='en', key='param_error', args=('invalid format',)")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_fallback_unknown_lang():
    """
    Lingua sconosciuta → fallback IT.
    Verifica che lingue non supportate fallback su italiano.
    """
    print("\n=== test_get_msg_fallback_unknown_lang ===")
    print("Testing: Lingua sconosciuta → fallback IT")
    
    result = get_msg("xx", "welcome")
    expected = MESSAGES["it"]["welcome"]
    
    print(f"Input: lang_code='xx', key='welcome'")
    print(f"Expected (IT fallback): {repr(expected[:50])}...")
    print(f"Actual:   {repr(result[:50])}...")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_get_msg_fallback_unknown_key():
    """
    Chiave sconosciuta → fallback EN.
    Verifica che chiavi non definite fallback su inglese.
    """
    print("\n=== test_get_msg_fallback_unknown_key ===")
    print("Testing: Chiave sconosciuta → fallback EN")
    
    result = get_msg("it", "unknown_key")
    expected = "unknown_key"
    
    print(f"Input: lang_code='it', key='unknown_key'")
    print(f"Expected (EN fallback): {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_all_keys_present_italian():
    """
    Tutte le chiavi presenti per italiano.
    Verifica che tutte le chiavi MESSAGES siano definite per italiano.
    """
    print("\n=== test_all_keys_present_italian ===")
    print("Testing: Tutte le chiavi presenti per italiano")
    
    keys = ["welcome", "cleaning", "cancel_btn", "cancel_msg", "general_error",
            "ask_zone", "no_data", "disambiguate", "oos_fallback", "param_error",
            "rate_limit", "invalid_id", "invalid_hour", "invalid_month", "internal_error"]
    
    missing = [k for k in keys if k not in MESSAGES.get("it", {})]
    
    print(f"Input: lang_code='it', keys={keys}")
    print(f"Missing keys: {missing if missing else 'None'}")
    print(f"Result: {'PASS' if not missing else 'FAIL'}")


def test_all_keys_present_english():
    """
    Tutte le chiavi presenti per inglese.
    Verifica che tutte le chiavi MESSAGES siano definite per inglese.
    """
    print("\n=== test_all_keys_present_english ===")
    print("Testing: Tutte le chiavi presenti per inglese")
    
    keys = ["welcome", "cleaning", "cancel_btn", "cancel_msg", "general_error",
            "ask_zone", "no_data", "disambiguate", "oos_fallback", "param_error",
            "rate_limit", "invalid_id", "invalid_hour", "invalid_month", "internal_error"]
    
    missing = [k for k in keys if k not in MESSAGES.get("en", {})]
    
    print(f"Input: lang_code='en', keys={keys}")
    print(f"Missing keys: {missing if missing else 'None'}")
    print(f"Result: {'PASS' if not missing else 'FAIL'}")


def test_language_code_partial():
    """
    Codice lingua parziale (it-IT, en-US).
    Verifica che codici lingua completi (it-IT, en-US) vengano trattati correttamente.
    """
    print("\n=== test_language_code_partial ===")
    print("Testing: Codice lingua parziale (it-IT, en-US)")
    
    result_it = get_msg("it-IT", "welcome")
    expected_it = MESSAGES["it"]["welcome"]
    
    result_en = get_msg("en-US", "welcome")
    expected_en = MESSAGES["en"]["welcome"]
    
    print(f"Input: lang_code='it-IT', key='welcome'")
    print(f"  Expected: {repr(expected_it[:30])}...")
    print(f"  Actual:   {repr(result_it[:30])}...")
    print(f"  Result: {'PASS' if result_it == expected_it else 'FAIL'}")
    
    print(f"Input: lang_code='en-US', key='welcome'")
    print(f"  Expected: {repr(expected_en[:30])}...")
    print(f"  Actual:   {repr(result_en[:30])}...")
    print(f"  Result: {'PASS' if result_en == expected_en else 'FAIL'}")


def test_msg_format_with_args():
    """
    Placeholder con argomenti multipli.
    Verifica che i placeholder multiple siano sostituiti correttamente.
    """
    print("\n=== test_msg_format_with_args ===")
    print("Testing: Placeholder con argomenti multipli")
    
    it_msg = MESSAGES["it"]["internal_error"]
    result = get_msg("it", "internal_error", "Database timeout")
    expected = it_msg.format("Database timeout")
    
    print(f"Input: lang_code='it', key='internal_error', args=('Database timeout',)")
    print(f"Expected: {repr(expected)}")
    print(f"Actual:   {repr(result)}")
    print(f"Result: {'PASS' if result == expected else 'FAIL'}")


def test_italian_keys_complete_set():
    """
    Set completo chiavi italiane.
    Verifica che il set di chiavi italiane sia completo.
    """
    print("\n=== test_italian_keys_complete_set ===")
    print("Testing: Set completo chiavi italiane")
    
    expected_keys = {"welcome", "cleaning", "cancel_btn", "cancel_msg", "general_error",
                    "ask_zone", "no_data", "disambiguate", "oos_fallback", "param_error",
                    "rate_limit", "invalid_id", "invalid_hour", "invalid_month", "internal_error"}
    actual_keys = set(MESSAGES.get("it", {}).keys())
    
    print(f"Expected keys: {sorted(expected_keys)}")
    print(f"Actual keys:   {sorted(actual_keys)}")
    print(f"Match: {'PASS' if expected_keys == actual_keys else 'FAIL'}")


def test_english_keys_complete_set():
    """
    Set completo chiavi inglesi.
    Verifica che il set di chiavi inglesi sia completo.
    """
    print("\n=== test_english_keys_complete_set ===")
    print("Testing: Set completo chiavi inglesi")
    
    expected_keys = {"welcome", "cleaning", "cancel_btn", "cancel_msg", "general_error",
                    "ask_zone", "no_data", "disambiguate", "oos_fallback", "param_error",
                    "rate_limit", "invalid_id", "invalid_hour", "invalid_month", "internal_error"}
    actual_keys = set(MESSAGES.get("en", {}).keys())
    
    print(f"Expected keys: {sorted(expected_keys)}")
    print(f"Actual keys:   {sorted(actual_keys)}")
    print(f"Match: {'PASS' if expected_keys == actual_keys else 'FAIL'}")


if __name__ == "__main__":
    test_get_msg_italian_welcome()
    test_get_msg_english_welcome()
    test_get_msg_italian_ask_zone()
    test_get_msg_english_ask_zone()
    test_get_msg_italian_oos_fallback()
    test_get_msg_english_oos_fallback()
    test_get_msg_italian_param_error()
    test_get_msg_english_param_error()
    test_get_msg_fallback_unknown_lang()
    test_get_msg_fallback_unknown_key()
    test_all_keys_present_italian()
    test_all_keys_present_english()
    test_language_code_partial()
    test_msg_format_with_args()
    test_italian_keys_complete_set()
    test_english_keys_complete_set()