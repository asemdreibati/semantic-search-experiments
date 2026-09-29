import requests

# ============================================================
# CONFIG
# ============================================================

SEARCH_URL = "http://34.196.113.253:8000/v1/query/search"

# TODO: put your real bearer token here
API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI2SS02UWMxbkkxQUt0c3R6UUNQRnFRSnN1cEREWDVCdHRiNHpMckhMUGJZIn0.eyJleHAiOjE4MjE0NDkyNjcsImlhdCI6MTc4OTkxMzI2NywianRpIjoiY2ZiMTliMjUtOWRkNS00NDVmLWIyNmEtN2E2MzkyMjkyNmNlIiwiaXNzIjoiaHR0cDovLzMyLjE5OS4yMzkuMTcwL2F1dGgvcmVhbG1zL3dhcmVkIiwic3ViIjoiNDQ3NTZjYTAtY2JlOC00YWRjLTkxNjQtNDgyZTE4NjUyYzhlIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiY3RzLXdlYiIsInNlc3Npb25fc3RhdGUiOiJmMjk3MGEwMC01NjI4LTQ4MWMtODQyNy05ZjIyZmQzOWIxYTIiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy13YXJlZC0xIl19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJmMjk3MGEwMC01NjI4LTQ4MWMtODQyNy05ZjIyZmQzOWIxYTIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsIm5hbWUiOiLZhdmI2LjZgSDYp9mE2KfYqti12KfZhNin2KoiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJrYWNzdC1jdHMiLCJsb2NhbGUiOiJhciIsImdpdmVuX25hbWUiOiLZhdmI2LjZgSDYp9mE2KfYqti12KfZhNin2KoiLCJmYW1pbHlfbmFtZSI6IiJ9.bCgNw_HCQUXq3v_NOk0IViNdAnPW6lw_JSAZyQN64iveN81o_q7UDFYiKJcnxI3FPxNLbAnp4Fu6BKr6ncAl61vIBfT07bHCV82P-2rRSDlhnMOVB9VNFmV921uqjLHgrTRurpR26U3YYRZjnDHUgMq24UBJjBemDtHiEGJCc3DQkAyMrGOkBr2-N8LJuORT9p4q75aKVWeFuRIfF3cLYSC6vsRVXrpZNfy81Axsij_TfrVUhgn5QDLuIuHx8nFJ0oZCOgpaV2Wfs81XnwY7EaAil7pVq6jEraoIVp8L46sH_jjMZIMe8FQ2OFMt2JWB0GWKxiZiJUP1Uuor9XPxXg"

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "accept": "application/json",
}

# Request body defaults (mirrors the curl example you shared)
DEFAULT_TOP_K = 8
DEFAULT_SIZE = 50
DEFAULT_MODE = "vector"
DEFAULT_RERANK = False
DEFAULT_EXPAND_WINDOW = 0
DEFAULT_UNDERSTAND = False


DOCUMENTS = {
    "862116af-472d-453f-9235-e8dc52ae0224":
        "لائحة الاتصالات الرسمية والمحافظة على الوثائق ومعلوماتها",

    "cb850770-7f27-4a47-b6cd-e456e9795734":
        "سياسة البيانات المفتوحة - جامعة الملك سعود",

    "d5245e7f-457b-4a15-844c-b36b8c2c7575":
        "قرار إعادة تشكيل مجلس الوزراء",

    "d6b2df42-9290-49ad-9be0-19ea6b7b0f72":
        "حوكمة التنسيق بين وزارة الصناعة والثروة المعدنية والجهات المختصة",
}

TARGET_DOCUMENTS = set(DOCUMENTS.keys())


TEST_CASES = [

    # ============================================================
    # Document 1
    # ============================================================

    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "تنظيم المراسلات",
        "query": "ضوابط توحيد شكل وطريقة التعامل مع المراسلات الرسمية",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "وضع الضوابط التي تنظم المراسلات الواردة والصادرة وتوثيقها، وتوحد معاييرها الموضوعية والشكلية",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "تنسيق الخطابات",
        "query": "طريقة كتابة بداية الخطاب الرسمي في أعلى الصفحة",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "تدون البسملة في منتصف أعلى الرسالة (بسم الله الرحمن الرحيم)",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "البريد الوارد",
        "query": "كيفية تقسيم سجلات المعاملات الواردة وترميزها",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "تقسم سجلات الوارد إلى ثلاثة سجلات، تعطى الرموز (٣،٢،١)",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "أهمية المراسلات",
        "query": "تمييز الخطابات التي تحتاج إلى اهتمام خاص عن طريق بطاقة ملونة",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "تميز موضوعات المراسلات المهمة - ذات القيمة - ببطاقة ذات لون ( أخضر ), وبدرجة أهمية واحدة ( مهم )",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "سرية المراسلات",
        "query": "من يستطيع فتح الخطابات السرية داخل الجهة",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "لا يفتح المراسلات السرية إلا من وجهت إليه الرسالة أو المختص الذي أذن له بفتحها",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "صلاحية التوقيع",
        "query": "من يملك صلاحية اعتماد وإصدار الخطاب الرسمي",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "أن تصدر بتوقيع صاحب الصلاحية أو من يفوضه",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "متابعة المعاملات",
        "query": "متابعة حركة المعاملات الواردة ومعرفة أين وصلت",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "تمكين المختصين بالمتابعة من الدخول على نظام الاتصالات الإدارية لمعرفة حركة المراسلات وفق صلاحيات محددة",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "حماية الوثائق",
        "query": "كيفية نقل الملفات والوثائق بطريقة تمنع تلفها أو كشف محتواها",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "أن تجري عملية النقل بطريقة آمنة يراعي فيها حفظ سرية المحتوى",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "الوثائق الدائمة",
        "query": "هل يمكن تداول النسخ الأصلية من الوثائق التي يجب الاحتفاظ بها بشكل دائم",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "لا يتم تداول أصول الوثائق الدائمة الحفظ، ويكتفى بتداول نسخ أو صور منها",
    },
    {
        "document": DOCUMENTS["862116af-472d-453f-9235-e8dc52ae0224"],
        "category": "نفاذ اللائحة",
        "query": "متى يبدأ تطبيق اللائحة بعد نشرها رسميا",
        "expected_uuid": "862116af-472d-453f-9235-e8dc52ae0224",
        "expected_text":
            "تنشر هذه اللائحة في الجريدة الرسمية، وتصبح نافذة بعد ( مائة وثمانين ) يوما من تاريخ نشرها",
    },

    # ============================================================
    # Document 2
    # ============================================================

    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "هدف السياسة",
        "query": "الغرض من تنظيم نشر البيانات العامة في الجامعة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "تهدف هذه السياسة إلى تحديد متطلبات نشر البيانات المفتوحة لجامعة الملك سعود - المصنفة بأنها بيانات عامة طبقا لسياسة تصنيف البيانات",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "نطاق التطبيق",
        "query": "ما نوع البيانات التي تشملها سياسة البيانات المفتوحة في الجامعة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "تطبق هذه السياسة على مجموعة محددة من البيانات والمعلومات العامة التي تنتجها أو تجمعها جامعة الملك سعود",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "البيانات العامة",
        "query": "البيانات التي يمكن نشرها باعتبارها معلومات عامة في الجامعة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "المصنفة بأنها بيانات عامة طبقا لسياسة تصنيف البيانات",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "التشريعات",
        "query": "الأنظمة والضوابط الوطنية التي تعتمد عليها سياسة نشر البيانات",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "الالتزام بالمتطلبات التشريعية الخاصة بها في وثيقتي «ضوابط إدارة البيانات وحوكمتها وحماية البيانات الشخصية",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "التقارير",
        "query": "التقرير السنوي الخاص بإنجازات خطة البيانات المفتوحة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "إعداد تقرير سنوي يقدم إلى مكتب إدارة البيانات الوطنية",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "مؤشرات الأداء",
        "query": "قياس تقدم الجامعة في تنفيذ خطة البيانات المفتوحة وعدد البيانات المنشورة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "الأهداف ومؤشرات الأداء الرئيسية المحددة في خطة البيانات المفتوحة",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "مجموعات البيانات",
        "query": "عدد مجموعات البيانات التي تم تجهيزها للنشر مقارنة بالمنشورة فعليا",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "عدد مجموعات البيانات المفتوحة المحددة للنشر. - عدد مجموعات البيانات المفتوحة المنشورة",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "التدقيق",
        "query": "الاستعداد لمراجعة الالتزام بسياسة البيانات المفتوحة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "رفع جاهزية الجامعة لأي عملية تدقيق داخلي أو خارجي لتقييم امتثالها لسياسة البيانات المفتوحة",
    },
    {
        "document": DOCUMENTS["cb850770-7f27-4a47-b6cd-e456e9795734"],
        "category": "عدم الامتثال",
        "query": "الإجراءات المتبعة عندما تظهر مخالفات أو نقاط عدم التزام بالسياسة",
        "expected_uuid": "cb850770-7f27-4a47-b6cd-e456e9795734",
        "expected_text":
            "عند وجود نقاط عدم امتثال، يقوم مكتب إدارة البيانات",
    },

    # ============================================================
    # Document 3
    # ============================================================

    {
        "document": DOCUMENTS["d5245e7f-457b-4a15-844c-b36b8c2c7575"],
        "category": "تشكيل مجلس الوزراء",
        "query": "القرار الملكي المتعلق بالتشكيل الجديد لمجلس الوزراء",
        "expected_uuid": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
        "expected_text":
            "وبعد الاطلاع على الأمرين الملكيين رقم (أ/٦١) ورقم (أ/٦٢) المؤرخين في ١٤٤٤/٣/١ه، الصادرين بشأن تشكيل مجلس الوزراء",
    },
    {
        "document": DOCUMENTS["d5245e7f-457b-4a15-844c-b36b8c2c7575"],
        "category": "المرجع النظامي",
        "query": "الأنظمة التي تم الرجوع إليها قبل إصدار قرار إعادة تشكيل الحكومة",
        "expected_uuid": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
        "expected_text":
            "بعد الاطلاع على النظام الأساسي للحكم، الصادر بالأمر الملكي رقم (أ/٩٠) بتاريخ ١٤١٢/٨/٢٧ه",
    },
    {
        "document": DOCUMENTS["d5245e7f-457b-4a15-844c-b36b8c2c7575"],
        "category": "مجلس الوزراء",
        "query": "المستند النظامي الذي ينظم عمل مجلس الوزراء",
        "expected_uuid": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
        "expected_text":
            "وبعد الاطلاع على المادة (٩) من نظام مجلس الوزراء",
    },
    {
        "document": DOCUMENTS["d5245e7f-457b-4a15-844c-b36b8c2c7575"],
        "category": "الأوامر الملكية",
        "query": "الأوامر السابقة المرتبطة بتشكيل مجلس الوزراء",
        "expected_uuid": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
        "expected_text":
            "الأمرين الملكيين رقم (أ/٦١) ورقم (أ/٦٢)",
    },
    {
        "document": DOCUMENTS["d5245e7f-457b-4a15-844c-b36b8c2c7575"],
        "category": "القرار",
        "query": "الوثيقة الرسمية التي تتناول إعادة تنظيم أعضاء مجلس الوزراء",
        "expected_uuid": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
        "expected_text":
            "في شأن تشكيل مجلس الوزراء",
    },

    # ============================================================
    # Document 4
    # ============================================================

    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "التعريفات",
        "query": "المقصود بحوكمة التعاون بين وزارة الصناعة والجهات الحكومية",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "الحوكمة: حوكمة التنسيق بين وزارة الصناعة والثروة المعدنية والجهات المختصة في شأن الإجراءات المتعلقة بالمنشآت الصناعية",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "المنشآت الصناعية",
        "query": "التنسيق الحكومي بشأن الإجراءات التي تؤثر على المصانع",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "في شأن الإجراءات المتعلقة بالمنشآت الصناعية",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "إيقاف الإنتاج",
        "query": "إجراءات الجهات الحكومية عندما يتم إيقاف خط إنتاج مصنع بسبب مخالفة",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "في حال أوقف خط الإنتاج بسبب ارتكاب مخالفة تستدعي ذلك نظاما؛ تزود الجهة المختصة الوزارة بتقرير تفصيلي عن أسباب الإيقاف",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "التقارير",
        "query": "المدة المحددة لإبلاغ وزارة الصناعة بأسباب إيقاف خط إنتاج",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "في مدة لا تزيد على (٥) أيام عمل من تاريخ إيقاف خط الإنتاج",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "تصحيح المخالفات",
        "query": "التنسيق مع المصنع لمعالجة المخالفة بعد إيقاف الإنتاج",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "تنسيق الوزارة مع المنشأة الصناعية لتصحيح المخالفة، وفق الإجراءات النظامية ذات الصلة",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "قياس الأثر",
        "query": "دراسة تأثير إغلاق خطوط الإنتاج على القطاع الصناعي ورؤية 2030",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "تقيس أثر ذلك على تطور الصناعة، ومستهدفات رؤية المملكة العربية السعودية ٢٠٣٠",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "اللجان التنسيقية",
        "query": "اجتماعات اللجان المشتركة بين الوزارة والجهات المختصة",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "تعقد اللجان التنسيقية اجتماعاتها -حضوريا أو مرئيا- متى دعت الحاجة إلى ذلك",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "توصيات اللجان",
        "query": "كيفية اعتماد قرارات وتوصيات اللجنة المشتركة",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "تصدر اللجان التنسيقية توصياتها بالاتفاق بين الوزارة والجهة المختصة ذات العلاقة",
    },
    {
        "document": DOCUMENTS["d6b2df42-9290-49ad-9be0-19ea6b7b0f72"],
        "category": "رفع النتائج",
        "query": "إلى من ترفع اللجنة نتائج اجتماعاتها والمعوقات التي تواجهها",
        "expected_uuid": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
        "expected_text":
            "ترفع كل لجنة تنسيقية نتائج أعمالها وتوصياتها واحتياجاتها وما قد يواجهها من معوقات إلى وزير الصناعة والثروة المعدنية، ووزير الجهة المختصة أو رئيسها",
    },

    # ============================================================
    # Negative / Garbage Tests
    # ============================================================

    {
        "document": "NEGATIVE",
        "category": "Garbage",
        "query": "طريقة تحضير القهوة العربية في المنزل",
        "expected_uuid": None,
        "expected_text": "لا يجب إرجاع أي من الوثائق الأربعة",
    },
    {
        "document": "NEGATIVE",
        "category": "Garbage",
        "query": "أفضل أماكن السياحة في أوروبا خلال فصل الصيف",
        "expected_uuid": None,
        "expected_text": "لا يجب إرجاع أي من الوثائق الأربعة",
    },
    {
        "document": "NEGATIVE",
        "category": "Garbage",
        "query": "كيفية إصلاح محرك السيارة إذا ارتفعت حرارته",
        "expected_uuid": None,
        "expected_text": "لا يجب إرجاع أي من الوثائق الأربعة",
    },
    {
        "document": "NEGATIVE",
        "category": "Garbage",
        "query": "تمارين رياضية لبناء العضلات في المنزل",
        "expected_uuid": None,
        "expected_text": "لا يجب إرجاع أي من الوثائق الأربعة",
    },
]


# ============================================================
# NEW: POST /v1/query/search request + response parsing
# ============================================================

def search_with_new_api(query_text,
                         top_k=DEFAULT_TOP_K,
                         size=DEFAULT_SIZE,
                         mode=DEFAULT_MODE,
                         rerank=DEFAULT_RERANK,
                         expand_window=DEFAULT_EXPAND_WINDOW,
                         understand=DEFAULT_UNDERSTAND):
    """
    Calls the new search endpoint:

        POST /v1/query/search
        { "query": ..., "topK": ..., "size": ..., "mode": ...,
          "rerank": ..., "expandWindow": ..., "understand": ... }

    Returns the raw JSON response (dict) so callers can inspect
    'hits', 'total', 'candidates', etc.
    """
    payload = {
        "query": query_text,
        "topK": top_k,
        "size": size,
        "mode": mode,
        "rerank": rerank,
        "expandWindow": expand_window,
        "understand": understand,
    }

    response = requests.post(
        SEARCH_URL,
        headers=HEADERS,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


def extract_hits(data):
    """Return the list of hit objects from a search response."""
    return data.get("hits", []) or []


def extract_doc_id(hit):
    """Pull the document id out of a hit's citation block."""
    if not hit:
        return None
    return hit.get("citation", {}).get("docId")


def extract_score(hit):
    if not hit:
        return None
    return hit.get("score")


def extract_subject(hit):
    if not hit:
        return None
    return hit.get("citation", {}).get("subject")


def run_build_mode():
    print("=" * 120)
    print("SEMANTIC SEARCH TEST CASES")
    print("=" * 120)

    for index, test in enumerate(TEST_CASES, start=1):
        print(f"\nTest Case #{index}")
        print(f"Document       : {test['document']}")
        print(f"Category       : {test['category']}")
        print(f"Query          : {test['query']}")
        print(f"Expected UUID  : {test['expected_uuid']}")
        print(f"Expected Match : {test['expected_text']}")


def run_verify_mode():
    passed = 0
    failed = 0

    print("=" * 140)
    print("SEMANTIC SEARCH QA (POST /v1/query/search)")
    print("=" * 140)

    for index, test in enumerate(TEST_CASES, start=1):

        try:
            data = search_with_new_api(test["query"])
            hits = extract_hits(data)

            top_hit = hits[0] if hits else None
            actual_doc_id = extract_doc_id(top_hit)
            actual_score = extract_score(top_hit)

            if test["expected_uuid"] is None:
                success = (
                    actual_doc_id is None
                    or actual_doc_id not in TARGET_DOCUMENTS
                )
            else:
                success = actual_doc_id == test["expected_uuid"]

            status = "PASS" if success else "FAIL"

            if success:
                passed += 1
            else:
                failed += 1

            print("\n" + "-" * 140)
            print(f"Test Case #{index}")
            print(f"Document       : {test['document']}")
            print(f"Category       : {test['category']}")
            print(f"Query          : {test['query']}")
            print(f"Expected UUID  : {test['expected_uuid']}")
            print(f"Expected Match : {test['expected_text']}")
            print(f"Actual DocId   : {actual_doc_id}")
            print(f"Actual Score   : {actual_score}")
            print(f"Returned Count : {len(hits)}")
            print(f"Total (server) : {data.get('total')}")
            print(f"Status         : {status}")

            if hits:
                print("Top Results:")
                for rank, hit in enumerate(hits[:5], start=1):
                    print(
                        f"  {rank}. "
                        f"score={extract_score(hit)} | "
                        f"docId={extract_doc_id(hit)} | "
                        f"subject={extract_subject(hit)}"
                    )

        except Exception as e:
            failed += 1

            print("\n" + "-" * 140)
            print(f"Test Case #{index}")
            print(f"Document       : {test['document']}")
            print(f"Category       : {test['category']}")
            print(f"Query          : {test['query']}")
            print(f"Expected UUID  : {test['expected_uuid']}")
            print(f"Expected Match : {test['expected_text']}")
            print("Actual DocId   : ERROR")
            print(f"Status         : FAIL")
            print(f"Error          : {e}")

    total = passed + failed

    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)
    print(f"Total Tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Pass Rate   : {(passed / total * 100):.2f}%" if total else "Pass Rate   : 0%")


if __name__ == "__main__":

    MODE = "verify"
    # MODE = "build"

    if MODE == "build":
        run_build_mode()
    elif MODE == "verify":
        run_verify_mode()
    else:
        raise ValueError("MODE must be 'build' or 'verify'")