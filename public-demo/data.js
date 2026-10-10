/* Authored synthetic documents. No benchmark pages, model outputs or private assets. */
globalThis.LAVA_DEMO_DATA = {
  schema: "lava-public-evidence-v1",
  evidence_type: "SYNTHETIC_ONLY",
  provenance: {
    author: "Alvaro Mendizabal",
    scope:
      "Authored Japanese and Vietnamese documents for an inspectable lexical retrieval and extractive evidence demonstration.",
    retrieval_source: "src/lava/retrieval/lexical.py",
    validation_source: "src/lava/readers/structured_output.py",
    model_inference: false,
    ocr_performed: false,
    private_assets_used: false,
    official_evaluation: false,
  },
  documents: [
    {
      id: "aoba-operations",
      language: "ja",
      language_label: "Japanese",
      code: "SYN · JA / 01",
      title: "青葉物流 業務報告",
      subtitle: "Quarterly operations · 5 physical pages",
      accent: "clay",
      description:
        "Shipments, service terms and storage conditions for an invented logistics company.",
      examples: [
        { label: "Tokyo shipments", query: "東京拠点の出荷注文数は？" },
        { label: "Return window", query: "返品の受付期限は？" },
        { label: "Storage temperature", query: "保管温度は？" },
        { label: "Unsupported query", query: "火星基地の建設費用は？" },
      ],
      pages: [
        {
          number: 1,
          printed_label: "Cover",
          status: "ok",
          section: "四半期レポート",
          title: "2025年 第4四半期",
          lines: [
            { id: "ja-1-a", text: "青葉物流 業務報告。" },
            {
              id: "ja-1-b",
              text: "対象期間は2025年10月1日から12月31日までです。",
            },
            {
              id: "ja-1-c",
              text: "本報告書には、拠点別の出荷実績、サービス条件、保管条件を記載しています。",
            },
            { id: "ja-1-d", text: "発行日は2026年1月15日です。" },
          ],
        },
        {
          number: 2,
          printed_label: "1",
          status: "ok",
          section: "01 / 出荷実績",
          title: "拠点別の出荷注文数",
          lines: [
            {
              id: "ja-2-a",
              label: "東京",
              text: "東京拠点の第4四半期の出荷注文数は1,200件です。",
            },
            {
              id: "ja-2-b",
              label: "大阪",
              text: "大阪拠点の第4四半期の出荷注文数は850件です。",
            },
            {
              id: "ja-2-c",
              label: "福岡",
              text: "福岡拠点の第4四半期の出荷注文数は640件です。",
            },
            {
              id: "ja-2-d",
              text: "件数は期間内に出荷が完了した注文を表し、キャンセルされた注文は含みません。",
            },
          ],
        },
        {
          number: 3,
          printed_label: "2",
          status: "ok",
          section: "02 / サービス条件",
          title: "返品とお問い合わせ",
          lines: [
            {
              id: "ja-3-a",
              label: "返品",
              text: "返品の受付期限は商品到着後14日以内です。",
            },
            {
              id: "ja-3-b",
              text: "返品の申請には、注文番号と商品の状態がわかる写真を添付してください。",
            },
            {
              id: "ja-3-c",
              label: "連絡",
              text: "お問い合わせの受付時間は平日の9時から17時までです。",
            },
            {
              id: "ja-3-d",
              text: "土曜日、日曜日、祝日はお問い合わせの受付を休止します。",
            },
          ],
        },
        {
          number: 4,
          printed_label: "3",
          status: "ok",
          section: "03 / 保管条件",
          title: "温度管理と記録",
          lines: [
            {
              id: "ja-4-a",
              label: "温度",
              text: "標準商品の保管温度は15℃から25℃までです。",
            },
            {
              id: "ja-4-b",
              label: "湿度",
              text: "保管場所の相対湿度は40％から60％を目安に管理します。",
            },
            { id: "ja-4-c", text: "温度と湿度の記録は毎日2回確認します。" },
            {
              id: "ja-4-d",
              text: "冷蔵品と冷凍品の保管条件は本報告書の対象外です。",
            },
          ],
        },
        {
          number: 5,
          printed_label: "Appendix A",
          status: "textless",
          section: "参考図版",
          title: "倉庫配置図",
          lines: [],
          display_note:
            "This authored diagram page has no indexed native text. The demo performs no OCR; its physical page number remains 5.",
        },
      ],
    },
    {
      id: "an-phuc-support",
      language: "vi",
      language_label: "Vietnamese",
      code: "SYN · VI / 02",
      title: "Thỏa thuận hỗ trợ An Phúc",
      subtitle: "Service agreement · 5 physical pages",
      accent: "teal",
      description:
        "Response times, service fees and data retention for an invented support provider.",
      examples: [
        {
          label: "Priority response",
          query: "Thời hạn phản hồi cho yêu cầu ưu tiên là bao lâu?",
        },
        {
          label: "Standard plan",
          query: "Gói Tiêu chuẩn có phí bao nhiêu mỗi tháng?",
        },
        {
          label: "Backup retention",
          query: "Bản sao lưu được giữ trong bao nhiêu ngày?",
        },
        { label: "Unsupported query", query: "quantum zeppelin orbit" },
      ],
      pages: [
        {
          number: 1,
          printed_label: "Bìa",
          status: "ok",
          section: "HỒ SƠ DỊCH VỤ",
          title: "Thỏa thuận hỗ trợ",
          lines: [
            { id: "vi-1-a", text: "Đơn vị cung cấp dịch vụ: An Phúc." },
            {
              id: "vi-1-b",
              text: "Thỏa thuận có hiệu lực từ ngày 1 tháng 1 năm 2026.",
            },
            {
              id: "vi-1-c",
              text: "Tài liệu mô tả thời hạn phản hồi, các gói dịch vụ và thời gian lưu trữ dữ liệu.",
            },
            {
              id: "vi-1-d",
              text: "Đây là tài liệu minh họa được soạn riêng cho bản trình diễn.",
            },
          ],
        },
        {
          number: 2,
          printed_label: "1",
          status: "ok",
          section: "01 / THỜI HẠN PHẢN HỒI",
          title: "Mức độ ưu tiên",
          lines: [
            {
              id: "vi-2-a",
              label: "Ưu tiên",
              text: "Yêu cầu ưu tiên được phản hồi trong vòng 4 giờ làm việc.",
            },
            {
              id: "vi-2-b",
              label: "Thông thường",
              text: "Yêu cầu thông thường được phản hồi trong vòng 24 giờ làm việc.",
            },
            {
              id: "vi-2-c",
              text: "Khung giờ hỗ trợ là 09:00–17:00, từ thứ Hai đến thứ Sáu.",
            },
            {
              id: "vi-2-d",
              text: "Yêu cầu được ghi nhận trong hệ thống hỗ trợ nội bộ.",
            },
          ],
        },
        {
          number: 3,
          printed_label: "2",
          status: "ok",
          section: "02 / CÁC GÓI DỊCH VỤ",
          title: "Phí dịch vụ hàng tháng",
          lines: [
            {
              id: "vi-3-a",
              label: "Tiêu chuẩn",
              text: "Gói Tiêu chuẩn có phí 1.200.000 đồng mỗi tháng.",
            },
            {
              id: "vi-3-b",
              label: "Nhóm",
              text: "Gói Nhóm có phí 2.800.000 đồng mỗi tháng.",
            },
            { id: "vi-3-c", text: "Các mức phí nêu trên chưa bao gồm thuế." },
            {
              id: "vi-3-d",
              text: "Khách hàng được chuyển gói vào ngày đầu tiên của tháng tiếp theo.",
            },
          ],
        },
        {
          number: 4,
          printed_label: "3",
          status: "ok",
          section: "03 / LƯU TRỮ DỮ LIỆU",
          title: "Sao lưu và nhật ký",
          lines: [
            {
              id: "vi-4-a",
              label: "Sao lưu",
              text: "Bản sao lưu được giữ trong 30 ngày.",
            },
            {
              id: "vi-4-b",
              label: "Nhật ký",
              text: "Nhật ký truy cập được giữ trong 90 ngày.",
            },
            {
              id: "vi-4-c",
              text: "Dữ liệu được xóa sau khi hết thời gian lưu trữ tương ứng.",
            },
            {
              id: "vi-4-d",
              text: "Chỉ nhân viên được phân quyền mới có thể xem nhật ký truy cập.",
            },
          ],
        },
        {
          number: 5,
          printed_label: "Phụ lục B",
          status: "extraction_error",
          section: "PHỤ LỤC",
          title: "Biểu đồ quy trình",
          lines: [],
          display_note:
            "A simulated extraction failure leaves this physical page unindexed. Unverified text is never supplied to retrieval or answer selection.",
        },
      ],
    },
  ],
};
