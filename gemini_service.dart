import 'dart:convert';
import 'dart:developer' as dev;
import 'package:flutter/foundation.dart'; // Added for compute (also exports typed_data)
import 'package:http/http.dart' as http;
import '../utils/base_url.dart';

/// Token counts from Gemini `usageMetadata` on each [generateContent] response.
/// - [promptTokenCount]: input (prompt + images, etc.)
/// - [candidatesTokenCount]: visible model output tokens (the JSON/text you receive)
/// - [thoughtsTokenCount]: internal reasoning tokens (Gemini 2.5+ “thinking”); **not**
///   included in [candidatesTokenCount], but **is** included in [totalTokenCount]
///   together with prompt and candidates.
/// - [totalTokenCount]: full billable total as reported by the API (typically
///   prompt + candidates + thoughts + other internal usage — do not expect
///   `total == input + output` when [thoughtsTokenCount] is non-zero).
/// - [cachedContentTokenCount]: tokens served from context cache when applicable
class GeminiTokenUsage {
  const GeminiTokenUsage({
    required this.promptTokenCount,
    required this.candidatesTokenCount,
    required this.totalTokenCount,
    this.cachedContentTokenCount = 0,
    this.thoughtsTokenCount = 0,
  });

  final int promptTokenCount;
  final int candidatesTokenCount;
  final int totalTokenCount;
  final int cachedContentTokenCount;

  /// Internal reasoning / “thinking” tokens (Gemini 2.5 family). Explains why
  /// [totalTokenCount] can be much larger than prompt + candidates alone.
  final int thoughtsTokenCount;

  static const GeminiTokenUsage zero = GeminiTokenUsage(
    promptTokenCount: 0,
    candidatesTokenCount: 0,
    totalTokenCount: 0,
    cachedContentTokenCount: 0,
    thoughtsTokenCount: 0,
  );

  factory GeminiTokenUsage.fromUsageMetadata(Map<String, dynamic>? m) {
    if (m == null) return GeminiTokenUsage.zero;
    int n(dynamic v) => (v is num) ? v.toInt() : int.tryParse('$v') ?? 0;
    return GeminiTokenUsage(
      promptTokenCount: n(m['promptTokenCount']),
      candidatesTokenCount: n(m['candidatesTokenCount']),
      totalTokenCount: n(m['totalTokenCount']),
      cachedContentTokenCount: n(m['cachedContentTokenCount']),
      thoughtsTokenCount: n(m['thoughtsTokenCount']),
    );
  }

  /// Sum usage across multiple API calls (e.g. JSON retry attempts).
  GeminiTokenUsage operator +(GeminiTokenUsage o) => GeminiTokenUsage(
    promptTokenCount: promptTokenCount + o.promptTokenCount,
    candidatesTokenCount: candidatesTokenCount + o.candidatesTokenCount,
    totalTokenCount: totalTokenCount + o.totalTokenCount,
    cachedContentTokenCount:
        cachedContentTokenCount + o.cachedContentTokenCount,
    thoughtsTokenCount: thoughtsTokenCount + o.thoughtsTokenCount,
  );

  /// Human-readable summary for logs/debug UI.
  String get displayLine =>
      'Input: $promptTokenCount tokens · Output: $candidatesTokenCount tokens'
      '${thoughtsTokenCount > 0 ? ' · Thinking: $thoughtsTokenCount tokens' : ''}'
      '${totalTokenCount > 0 ? ' · Total: $totalTokenCount' : ''}'
      '${cachedContentTokenCount > 0 ? ' · Cached: $cachedContentTokenCount' : ''}';

  /// True if the API reported any non-zero usage field.
  bool get hasAnyCount =>
      promptTokenCount > 0 ||
      candidatesTokenCount > 0 ||
      totalTokenCount > 0 ||
      cachedContentTokenCount > 0 ||
      thoughtsTokenCount > 0;

  /// Reads [usageMetadata] from a generateContent JSON response map.
  static GeminiTokenUsage? usageFromDecoded(Map<String, dynamic> decoded) {
    final raw = decoded['usageMetadata'];
    if (raw == null) return null;
    if (raw is Map<String, dynamic>) {
      return GeminiTokenUsage.fromUsageMetadata(raw);
    }
    if (raw is Map) {
      return GeminiTokenUsage.fromUsageMetadata(Map<String, dynamic>.from(raw));
    }
    return null;
  }
}

/// One successful [generateContent] response: model text + optional usage.
class GeminiGenerateResult {
  GeminiGenerateResult({required this.text, this.tokenUsage});
  final String text;
  final GeminiTokenUsage? tokenUsage;
}

class TeacherAnnotation {
  final int pageIndex; // 0-indexed
  final double yPositionPercent; // 0-100, 0=top, 100=bottom
  final double xStartPercent; // 0-100, horizontal start of underline
  final double xEndPercent; // 0-100, horizontal end of underline
  final String comment;
  final bool isPositive;
  final String lineStyle; // "straight" or "zigzag"

  TeacherAnnotation({
    required this.pageIndex,
    required this.yPositionPercent,
    required this.xStartPercent,
    required this.xEndPercent,
    required this.comment,
    required this.isPositive,
    required this.lineStyle,
  });

  TeacherAnnotation copyWith({
    int? pageIndex,
    double? yPositionPercent,
    double? xStartPercent,
    double? xEndPercent,
    String? comment,
    bool? isPositive,
    String? lineStyle,
  }) {
    return TeacherAnnotation(
      pageIndex: pageIndex ?? this.pageIndex,
      yPositionPercent: yPositionPercent ?? this.yPositionPercent,
      xStartPercent: xStartPercent ?? this.xStartPercent,
      xEndPercent: xEndPercent ?? this.xEndPercent,
      comment: comment ?? this.comment,
      isPositive: isPositive ?? this.isPositive,
      lineStyle: lineStyle ?? this.lineStyle,
    );
  }
}

class GeminiAnalysisOutput {
  final String studentText;
  final double marksAwarded;
  final double confidencePercent;
  final String goodPoints;
  final String improvements;
  final String finalReview;
  final List<TeacherAnnotation> annotations;

  /// Populated from API `usageMetadata` (sums retries if multiple calls).
  final GeminiTokenUsage? tokenUsage;

  GeminiAnalysisOutput({
    required this.studentText,
    required this.marksAwarded,
    required this.confidencePercent,
    required this.goodPoints,
    required this.improvements,
    required this.finalReview,
    required this.annotations,
    this.tokenUsage,
  });
}

/// One cell in the intro-page marks table extracted by AI.
/// [questionNo] == 0 means the "Total" row.
class IntroMarkCell {
  final int questionNo;
  final String marksText; // exactly as written, e.g. "4.00", "3-75", "91"
  final double xPercent; // horizontal centre of value on page (0–100)
  final double yPercent; // vertical centre of value on page (0–100)

  const IntroMarkCell({
    required this.questionNo,
    required this.marksText,
    required this.xPercent,
    required this.yPercent,
  });
}

/// Aggregates all extracted marks from the intro/cover page.
class IntroPageAnalysis {
  final List<IntroMarkCell> cells;
  final GeminiTokenUsage? tokenUsage;
  const IntroPageAnalysis({required this.cells, this.tokenUsage});
}

/// Holds the AI-synthesised combined key review for a full session.
class CombinedKeyReviewOutput {
  final String overallImprovements;
  final String oneThingToWrite;
  final String overallReview;
  final GeminiTokenUsage? tokenUsage;

  const CombinedKeyReviewOutput({
    required this.overallImprovements,
    required this.oneThingToWrite,
    required this.overallReview,
    this.tokenUsage,
  });
}

class GeminiService {
  static const _model = 'gemini-2.5-flash';

  static String get _url =>
      'https://generativelanguage.googleapis.com/v1beta/models/$_model:generateContent?key=$giminiKey';

  /// Runs the analysis in a background isolate to prevent UI freezing
  /// during heavy base64 image encoding and JSON parsing.
  static Future<GeminiAnalysisOutput> analyseInIsolate({
    required List<Uint8List> pageImages,
    required String questionTitle,
    String? instructionName,
    required String modelDescription,
    required int totalMarks,
    String language = 'en',
    String checkLevel = 'Moderate',
  }) async {
    return compute(_analyseIsolateHandler, {
      'pageImages': pageImages,
      'questionTitle': questionTitle,
      'instructionName': instructionName,
      'modelDescription': modelDescription,
      'totalMarks': totalMarks,
      'language': language,
      'checkLevel': checkLevel,
    });
  }

  /// Re-analyses using already-extracted OCR text — skips sending images again.
  /// Preserves all annotations / grading logic but avoids re-OCR cost.
  static Future<GeminiAnalysisOutput> analyseWithCachedOcr({
    required String cachedStudentText,
    required String questionTitle,
    String? instructionName,
    required String modelDescription,
    required int totalMarks,
    required int pageCount,
    String language = 'en',
    String checkLevel = 'Moderate',
  }) async {
    return compute(_analyseWithCachedOcrHandler, {
      'cachedStudentText': cachedStudentText,
      'questionTitle': questionTitle,
      'instructionName': instructionName,
      'modelDescription': modelDescription,
      'totalMarks': totalMarks,
      'pageCount': pageCount,
      'language': language,
      'checkLevel': checkLevel,
    });
  }

  // ── Combined Key Review ─────────────────────────────────────────────────

  /// Takes per-question analysis results for one full student PDF session and
  /// asks Gemini to synthesise a single, coherent "combined key review" — the
  /// same content that appears as the handwritten end-page in the downloaded PDF.
  ///
  /// [questionResults] is a list of maps with keys:
  ///   questionNo, title, marksAwarded, marksTotal, improvements,
  ///   goodPoints, finalReview.
  static Future<CombinedKeyReviewOutput> generateCombinedKeyReview({
    required List<Map<String, dynamic>> questionResults,
  }) async {
    return compute(_combinedKeyReviewHandler, {
      'questionResults': questionResults,
    });
  }

  // ── Intro-page analysis ─────────────────────────────────────────────────

  /// Sends the first intro page to Gemini to extract the teacher's handwritten
  /// marks from the M.Obt. (Marks Obtained) column plus their page positions.
  /// Runs in a background isolate — safe to call from the UI thread.
  static Future<IntroPageAnalysis> analyseIntroPageInIsolate(
    Uint8List pageImage,
  ) async {
    return compute(_analyseIntroPageHandler, {'pageImage': pageImage});
  }

  static Future<IntroPageAnalysis> _doAnalyseIntroPage(
    Uint8List pageImage,
  ) async {
    final base64Image = base64Encode(pageImage);

    // Pipe-delimited format is immune to JSON escaping / truncation bugs.
    // Each output line: questionNo|marksText|xPercent|yPercent
    // questionNo 0 = Total row.  marksText is empty string if cell is blank.
    const prompt = '''
You are analysing the COVER / INTRO page of a student exam answer sheet.

The page contains a MARKS TABLE with columns: Q.No. | M.Mark | M.Obt.

TASK:
1. Find the M.Obt. column (where the teacher writes the marks obtained).
2. For EVERY row in that table — question rows AND the Total row — output one line in this exact format:
   questionNo|marksText|xPercent|yPercent
   - questionNo : integer (use 0 for the Total/Grand-Total row)
   - marksText  : the handwritten number if visible, otherwise leave it empty (nothing between the pipes)
   - xPercent   : horizontal centre of the M.Obt. cell as % of page width (0–100)
   - yPercent   : vertical centre of the M.Obt. cell as % of page height (0–100)
3. Include ALL rows, even if M.Obt. cell is empty.
4. Output ONLY the data lines. No headers, no explanation, no JSON, no markdown.

Example (values are illustrative only — use real values from the image):
1|4|74|30
2||74|33
3|3.5|74|36
0|91|74|88
''';

    final body = jsonEncode({
      'contents': [
        {
          'parts': [
            {'text': prompt},
            {
              'inline_data': {'mime_type': 'image/jpeg', 'data': base64Image},
            },
          ],
        },
      ],
      'generationConfig': {'maxOutputTokens': 2048, 'temperature': 0.0},
    });

    final response = await http.post(
      Uri.parse(_url),
      headers: {'Content-Type': 'application/json'},
      body: body,
    );

    if (response.statusCode != 200) {
      dev.log(
        '[GeminiService] IntroPage HTTP ${response.statusCode}: ${response.body}',
        name: 'GeminiService',
      );
      return const IntroPageAnalysis(cells: []);
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final tokenUsage = GeminiTokenUsage.usageFromDecoded(decoded);
    if (tokenUsage != null && tokenUsage.hasAnyCount) {
      dev.log(
        '[GeminiService] IntroPage → ${tokenUsage.displayLine}',
        name: 'GeminiService',
      );
    }

    final candidates = decoded['candidates'] as List? ?? [];
    if (candidates.isEmpty) {
      return IntroPageAnalysis(cells: [], tokenUsage: tokenUsage);
    }

    final parts = (candidates[0]['content'] as Map)['parts'] as List;
    final rawText = (parts[0]['text'] as String? ?? '').trim();

    // Parse pipe-delimited lines: questionNo|marksText|xPercent|yPercent
    final cells = <IntroMarkCell>[];
    for (final line in rawText.split('\n')) {
      final trimmed = line.trim();
      if (trimmed.isEmpty || !trimmed.contains('|')) continue;
      final cols = trimmed.split('|');
      if (cols.length < 4) continue;
      final qNo = int.tryParse(cols[0].trim());
      if (qNo == null) continue;
      final marks = cols[1].trim();
      final xPct = double.tryParse(cols[2].trim()) ?? 50.0;
      final yPct = double.tryParse(cols[3].trim()) ?? 50.0;
      cells.add(
        IntroMarkCell(
          questionNo: qNo,
          marksText: marks,
          xPercent: xPct,
          yPercent: yPct,
        ),
      );
    }

    if (cells.isEmpty) {
      // Last-resort: fall back to regex extraction on the raw text.
      dev.log(
        '[GeminiService] IntroPage pipe parse got 0 cells — trying regex fallback',
        name: 'GeminiService',
      );
      return IntroPageAnalysis(
        cells: _regexExtractIntroCells(rawText),
        tokenUsage: tokenUsage,
      );
    }

    return IntroPageAnalysis(cells: cells, tokenUsage: tokenUsage);
  }

  /// Last-resort intro-cells extractor from malformed JSON-like output.
  static List<IntroMarkCell> _regexExtractIntroCells(String text) {
    final cells = <IntroMarkCell>[];
    final re = RegExp(
      "(?:questionNo|qNo)\\s*['\\\"]?\\s*:\\s*([0-9]+).*?"
      "(?:marksText|marks)\\s*['\\\"]?\\s*:\\s*['\\\"]?([^,\\\"'}\\n\\r]*)['\\\"]?.*?"
      "xPercent\\s*['\\\"]?\\s*:\\s*([0-9]+(?:\\.[0-9]+)?).*?"
      "yPercent\\s*['\\\"]?\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)",
      dotAll: true,
      caseSensitive: false,
    );
    for (final m in re.allMatches(text)) {
      cells.add(
        IntroMarkCell(
          questionNo: int.tryParse(m.group(1) ?? '0') ?? 0,
          marksText: (m.group(2) ?? '').trim(),
          xPercent: double.tryParse(m.group(3) ?? '50') ?? 50.0,
          yPercent: double.tryParse(m.group(4) ?? '50') ?? 50.0,
        ),
      );
    }
    return cells;
  }

  static Future<CombinedKeyReviewOutput> _doCombinedKeyReview({
    required List<Map<String, dynamic>> questionResults,
  }) async {
    final buffer = StringBuffer();
    for (int i = 0; i < questionResults.length; i++) {
      final q = questionResults[i];
      buffer.writeln('--- Question ${q['questionNo']} : ${q['title']} ---');
      buffer.writeln('Marks: ${q['marksAwarded']} / ${q['marksTotal']}');
      if ((q['goodPoints'] as String).isNotEmpty) {
        buffer.writeln('Good Points: ${q['goodPoints']}');
      }
      if ((q['improvements'] as String).isNotEmpty) {
        buffer.writeln('Improvements: ${q['improvements']}');
      }
      if ((q['finalReview'] as String).isNotEmpty) {
        buffer.writeln('Final Review: ${q['finalReview']}');
      }
      buffer.writeln();
    }

    final questionSummary = buffer.toString().trim();

    final prompt =
        '''
You are an experienced school teacher writing a detailed end-of-paper comment for a student.
Below are the per-question analysis results:

$questionSummary

YOUR TASK:
Write a "finalReview" that is a flowing, natural paragraph-style teacher comment of AT LEAST 150 words (aim for 180-220 words). It should read exactly like a real teacher's handwritten remark at the end of a corrected answer sheet — warm, personal, specific, and professional.

Structure (all in one block of plain prose, no headings, no bullets, no numbering, no markdown):
1. Start with 2-3 sentences acknowledging what the student did well across the paper, mentioning specific questions or topics.
2. Write 3-4 sentences identifying the most important weaknesses, with concrete examples from the student's answers (e.g. "In Q3 you left the conclusion incomplete…").
3. Give 2-3 sentences of clear, actionable advice on how to improve — specific study tips or practice habits.
4. End with 1-2 encouraging sentences motivating the student to keep working hard.

Keep every sentence natural and conversational — the way a teacher actually writes, not like a report. Do NOT use any symbols, asterisks, or special characters.

Also return:
- "overallImprovements": 4 plain sentences of improvement points (one per line, no symbols).
- "oneThingToWrite": ONE sentence — the single most impactful practice tip.

LANGUAGE: Match the dominant language in the question improvements above (Hindi or English).

Return ONLY valid JSON, no markdown:
{
  "finalReview": "<flowing paragraph of 150-220 words, sentences separated by \\n>",
  "overallImprovements": "<4 lines separated by \\n>",
  "oneThingToWrite": "<one sentence>"
}
''';

    final body = {
      'contents': [
        {
          'parts': [
            {'text': prompt},
          ],
        },
      ],
      'generationConfig': {
        'temperature': 0.35,
        'maxOutputTokens': 2048,
        'responseMimeType': 'application/json',
      },
    };

    dev.log(
      '[GeminiService] CombinedKeyReview → ${questionResults.length} question(s)',
      name: 'GeminiService',
    );

    // Retry loop: up to 3 attempts for a clean JSON response.
    const maxAttempts = 3;
    String? lastCleaned;
    Map<String, dynamic>? data;
    var usageAcc = GeminiTokenUsage.zero;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      final gen = await _callGeminiRaw(body, label: 'CombinedReview');
      lastCleaned = gen.text;
      usageAcc = usageAcc + (gen.tokenUsage ?? GeminiTokenUsage.zero);
      try {
        data = jsonDecode(gen.text) as Map<String, dynamic>;
        if (attempt > 1) {
          dev.log(
            '[GeminiService] CombinedKeyReview JSON OK on retry $attempt.',
            name: 'GeminiService',
          );
        }
        break;
      } catch (e) {
        dev.log(
          '[GeminiService] CombinedKeyReview JSON parse failed (attempt $attempt/$maxAttempts): $e',
          name: 'GeminiService',
        );
        if (attempt < maxAttempts) {
          await Future.delayed(Duration(milliseconds: 300 * attempt));
        }
      }
    }

    // All retries exhausted — repair then regex-fallback on the last response.
    if (data == null) {
      final cleaned = lastCleaned ?? '{}';
      final repaired = _repairTruncatedJson(cleaned);
      try {
        data = jsonDecode(repaired) as Map<String, dynamic>;
        dev.log(
          '[GeminiService] CombinedKeyReview JSON repaired after retries.',
          name: 'GeminiService',
        );
      } catch (_) {
        dev.log(
          '[GeminiService] CombinedKeyReview repair failed — using regex fallback.',
          name: 'GeminiService',
        );
        data = _regexExtractCombinedReviewFields(cleaned);
      }
    }

    String parseField(dynamic val) {
      if (val is List) return val.join('\n');
      return val?.toString() ?? '';
    }

    // Prefer the new compact "finalReview" field; fall back to "overallReview".
    final finalReview = parseField(data['finalReview']).trim();
    final legacyReview = parseField(data['overallReview']).trim();
    final chosenReview = finalReview.isNotEmpty ? finalReview : legacyReview;

    return CombinedKeyReviewOutput(
      overallImprovements: parseField(data['overallImprovements']),
      oneThingToWrite: parseField(data['oneThingToWrite']),
      overallReview: chosenReview,
      tokenUsage: usageAcc.hasAnyCount ? usageAcc : null,
    );
  }

  /// Sends [body] to the Gemini API and returns the raw text from the first
  /// candidate's first part.  Throws on non-200 or empty candidates.
  /// Does NOT parse JSON — callers handle that so they can retry.
  static Future<GeminiGenerateResult> _callGeminiRaw(
    Map<String, dynamic> body, {
    String label = 'Analysis',
  }) async {
    final response = await http.post(
      Uri.parse(_url),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (response.statusCode != 200) {
      throw Exception(
        'Gemini API error (${response.statusCode}): ${response.body}',
      );
    }
    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final tokenUsage = GeminiTokenUsage.usageFromDecoded(decoded);
    if (tokenUsage != null && tokenUsage.hasAnyCount) {
      dev.log(
        '[GeminiService] $label → ${tokenUsage.displayLine}',
        name: 'GeminiService',
      );
    }

    final candidates = decoded['candidates'] as List?;
    if (candidates == null || candidates.isEmpty) {
      throw Exception('Gemini returned no candidates.');
    }
    final content = candidates[0]['content'] as Map<String, dynamic>;
    final parts = content['parts'] as List;
    final raw = parts[0]['text'] as String? ?? '{}';
    final text = raw
        .replaceAll(RegExp(r'```json\s*'), '')
        .replaceAll(RegExp(r'```\s*'), '')
        .trim();
    return GeminiGenerateResult(text: text, tokenUsage: tokenUsage);
  }

  /// Closes an unterminated JSON string and any unclosed braces so the
  /// result can be parsed despite Gemini truncating the response.
  static String _repairTruncatedJson(String raw) {
    var text = raw.trimRight();

    // Walk the string tracking whether we are inside a JSON string value,
    // so we know exactly what needs to be closed.
    bool inString = false;
    bool escaped = false;
    int openBraces = 0;
    int openBrackets = 0;

    for (int i = 0; i < text.length; i++) {
      final c = text[i];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (c == r'\') {
        escaped = true;
        continue;
      }
      if (c == '"') {
        inString = !inString;
        continue;
      }
      if (!inString) {
        if (c == '{') {
          openBraces++;
        } else if (c == '}') {
          openBraces = (openBraces - 1).clamp(0, 999);
        } else if (c == '[') {
          openBrackets++;
        } else if (c == ']') {
          openBrackets = (openBrackets - 1).clamp(0, 999);
        }
      }
    }

    final buf = StringBuffer(text);
    // Close unterminated string value.
    if (inString) buf.write('"');
    // Remove trailing comma before closing.
    var result = buf.toString().trimRight();
    if (result.endsWith(',')) result = result.substring(0, result.length - 1);
    // Close unclosed arrays first, then braces.
    final closers = StringBuffer(result);
    for (int i = 0; i < openBrackets; i++) {
      closers.write(']');
    }
    for (int i = 0; i < openBraces; i++) {
      closers.write('}');
    }
    return closers.toString();
  }

  /// Parses the analysis JSON produced by both `analyse` and
  /// `_analyseFromCachedText`.  On parse failure it attempts repair, then
  /// returns whatever could be recovered.  Incomplete annotation objects
  /// (missing required fields) are silently dropped rather than crashing.
  static Map<String, dynamic> _parseAnalysisJson(String cleaned) {
    // 1. Try direct parse.
    try {
      return jsonDecode(cleaned) as Map<String, dynamic>;
    } catch (firstError) {
      dev.log(
        '[GeminiService] Analysis JSON parse failed — attempting repair. $firstError',
        name: 'GeminiService',
      );
    }

    // 2. Repair truncated JSON then re-parse.
    try {
      final repaired = _repairTruncatedJson(cleaned);
      final data = jsonDecode(repaired) as Map<String, dynamic>;
      dev.log(
        '[GeminiService] Analysis JSON repaired OK.',
        name: 'GeminiService',
      );
      // Scrub any annotation objects that are incomplete (truncation artefacts).
      if (data['annotations'] is List) {
        data['annotations'] = (data['annotations'] as List).where((a) {
          if (a is! Map) return false;
          return a.containsKey('xEndPercent') &&
              a.containsKey('comment') &&
              a.containsKey('pageIndex');
        }).toList();
      }
      return data;
    } catch (repairError) {
      dev.log(
        '[GeminiService] Analysis JSON repair also failed — using partial regex fallback. $repairError',
        name: 'GeminiService',
      );
    }

    // 3. Last-resort: extract scalar fields and individual annotation objects via
    //    regex so that even a severely truncated response gives useful output.
    final result = <String, dynamic>{
      'studentText': '',
      'marksAwarded': 0.0,
      'confidencePercent': 0.0,
      'goodPoints': '',
      'improvements': '',
      'finalReview': '',
      'annotations': <dynamic>[],
    };
    for (final field in [
      'studentText',
      'goodPoints',
      'improvements',
      'finalReview',
    ]) {
      final re = RegExp(
        '"$field"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)',
        dotAll: true,
      );
      final m = re.firstMatch(cleaned);
      if (m != null) {
        result[field] =
            m
                .group(1)
                ?.replaceAll(r'\n', '\n')
                .replaceAll(r'\"', '"')
                .replaceAll(r'\\', r'\')
                .trim() ??
            '';
      }
    }
    final marksRe = RegExp(r'"marksAwarded"\s*:\s*([0-9]+(?:\.[0-9]+)?)');
    final marksM = marksRe.firstMatch(cleaned);
    if (marksM != null)
      result['marksAwarded'] = double.tryParse(marksM.group(1) ?? '0') ?? 0.0;
    final confRe = RegExp(r'"confidencePercent"\s*:\s*([0-9]+(?:\.[0-9]+)?)');
    final confM = confRe.firstMatch(cleaned);
    if (confM != null)
      result['confidencePercent'] =
          double.tryParse(confM.group(1) ?? '0') ?? 0.0;

    // Try to salvage individual annotation objects from the broken JSON.
    // Each annotation block starts with "pageIndex": and contains a "comment":
    final annBlockRe = RegExp(
      r'\{[^{}]*"pageIndex"\s*:\s*(\d+)[^{}]*"yPositionPercent"\s*:\s*([0-9.]+)'
      r'[^{}]*"xStartPercent"\s*:\s*([0-9.]+)[^{}]*"xEndPercent"\s*:\s*([0-9.]+)'
      r'[^{}]*"comment"\s*:\s*"((?:[^"\\]|\\.)*)"'
      r'[^{}]*"isPositive"\s*:\s*(true|false)',
      dotAll: true,
    );
    final salvaged = <dynamic>[];
    for (final m in annBlockRe.allMatches(cleaned)) {
      salvaged.add({
        'pageIndex': int.tryParse(m.group(1) ?? '0') ?? 0,
        'yPositionPercent': double.tryParse(m.group(2) ?? '50') ?? 50.0,
        'xStartPercent': double.tryParse(m.group(3) ?? '20') ?? 20.0,
        'xEndPercent': double.tryParse(m.group(4) ?? '80') ?? 80.0,
        'comment': (m.group(5) ?? '')
            .replaceAll(r'\"', '"')
            .replaceAll(r'\n', '\n'),
        'isPositive': m.group(6) == 'true',
        'lineStyle': 'straight',
      });
    }
    if (salvaged.isNotEmpty) result['annotations'] = salvaged;

    return result;
  }

  /// Last-resort field extractor: reads each known key with a regex so we
  /// still get something useful from severely truncated JSON.
  static Map<String, dynamic> _regexExtractCombinedReviewFields(String text) {
    final result = <String, dynamic>{};
    for (final field in [
      'finalReview',
      'overallImprovements',
      'oneThingToWrite',
      'overallReview',
    ]) {
      // Matches "field": "value" where value may contain escaped chars.
      final re = RegExp(
        '"$field"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)',
        dotAll: true,
      );
      final m = re.firstMatch(text);
      if (m != null) {
        result[field] =
            m
                .group(1)
                ?.replaceAll(r'\n', '\n')
                .replaceAll(r'\"', '"')
                .replaceAll(r'\\', r'\')
                .trim() ??
            '';
      } else {
        result[field] = '';
      }
    }
    return result;
  }

  // ── Blank page detection ─────────────────────────────────────────────────

  static Future<GeminiAnalysisOutput> _analyseFromCachedText({
    required String cachedStudentText,
    required String questionTitle,
    String? instructionName,
    required String modelDescription,
    required int totalMarks,
    required int pageCount,
    String language = 'en',
    String checkLevel = 'Moderate',
  }) async {
    final checkLevelInstruction = checkLevel.toLowerCase() == 'hard'
        ? '- EVALUATION STRICTNESS: HARD. Be extremely strict. All answers must be strictly evaluated and normally score less than 50% of the total marks unless they are absolutely perfect without any flaws.'
        : '- EVALUATION STRICTNESS: MODERATE. Grade normally, but keep medium or average answers around or below 50% of the total marks.';

    final prompt =
        '''
You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

The student's handwritten answer has already been transcribed for you (OCR result):
"""
$cachedStudentText
"""

${instructionName != null && instructionName.isNotEmpty ? 'EXTRA ANSWER INSTRUCTIONS:\n$instructionName\n\n' : ''}QUESTION TITLE:
$questionTitle

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
$modelDescription

TOTAL MARKS FOR THIS QUESTION: $totalMarks

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
$checkLevelInstruction
- Award marksAwarded as a DECIMAL in multiples of 0.5 (e.g. 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 …). NEVER exceed $totalMarks.
- Marking tiers by question size:
    8-mark question  → Bahot achha (excellent): 3.5  |  Moderate: 2.5  |  Low: 1.5
    12-mark question → Bahot achha (excellent): 5    |  Moderate: 3    |  Low: 2.5
    Other totals     → scale proportionally using 0.5-step decimals.
- You are a STRICT examiner. Default to FEWER marks, not more.
- FULL marks ONLY if answer nearly perfectly matches the scheme — all key points, correct terminology, clear reasoning. This is rare.
- Do NOT reward effort or length — reward accuracy and relevance only.
- A long but mostly irrelevant answer scores LOW. A short precise answer can outscore a long vague one.
- Do NOT give benefit of the doubt. If a key point is not clearly stated, do not assume it was implied.
- Deduct for: wrong facts, missing key terms, no examples when required, incorrect conclusions.
- When in doubt between two values, always choose the LOWER one.

YOUR TASKS:

1. GRADE objectively against the marking scheme based on the transcribed text above.
3. ANNOTATE the answer: mark 2–5 specific spots with short teacher-style comments. Estimate page/position from the text structure (the answer spans $pageCount page(s)).

ANNOTATION PLACEMENT RULES (CRITICAL — prevents overlapping comments):
- On any single page, every annotation MUST have a yPositionPercent that differs by AT LEAST 20 from every other annotation ON THE SAME SIDE (left or right). Space them out evenly across the page height (e.g. 10, 30, 50, 70, 90).
- Two annotations that would be on the SAME side and within 20% of each other vertically → keep only the more important one and DISCARD the other.
- Two annotations that are on OPPOSITE sides (one left half, one right half) CAN share a similar yPositionPercent — this is fine and encouraged to keep them visually spread.
- Decide left vs right using xStartPercent and xEndPercent: if midX = (xStart+xEnd)/2 < 50, place it on the LEFT half (xEndPercent ≤ 50); otherwise on the RIGHT half (xStartPercent ≥ 50). Strictly alternate left/right when possible.
- Prefer spreading annotations across different pages — do not cluster all on page 0.
- Aim for at most 2–3 annotations per page. If you have more, move the extras to other pages or drop the least important ones.
- IMPORTANT: ALL annotations MUST have "isPositive": true. Only mark things worth circling positively (correct facts, good phrases, relevant points). Do NOT mark errors with annotations — errors should be mentioned in the "improvements" field instead.

TONE & LANGUAGE GUIDELINES:
Write ALL feedback as a professional yet approachable teacher — clear, constructive, specific.

For goodPoints: Address the student directly and be specific about what was done well.
For improvements: Be specific — mention what was missed and why it matters for marks.
For finalReview: Write 2–3 sentences as a professional remark — warm, personal, constructive.

${language == 'hi' ? '''LANGUAGE INSTRUCTION:
This is a Hindi-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in Hindi. No English feedback text at all.

HINDI VOCABULARY RULES (apply strictly):
- Use "saraahaniyan" or "Utkrisht" for praise — NOT just "achha".
- Use "Prayas" when noting effort.
- Nishkarsh (conclusion) must be future-oriented: "Aapka nishkarsh bhavishya unmukhi hona chahiye."
- Address the student as "Aap" / "Aapka" — NEVER "Tumne" or "Tu".
- "Introduction" → write "Prashtavana" or "Parichay" — never the English word.
- NEVER use the word "Shabash".
- "Utpatti" → use "Prashtuti".
- Strong conclusion: "Aapka nishkarsh prabhavshali hai."
- Line could be more specific: "Yah line aur vishishth ho sakti hai; udaharan ke sath samjhaya ja sakta tha."
- Replace "Sahi dhang se samjhaya hai" with "Sahi dhang se prastut kiya hai."''' : '''LANGUAGE INSTRUCTION:
This is an English-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in English. No Hindi feedback text at all.'''}

══════════════════════════════════════════
OUTPUT FORMAT:
══════════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no explanation outside JSON) with exactly these keys:
{
  "studentText": "$cachedStudentText",
  "marksAwarded": <decimal in multiples of 0.5, range 0 to $totalMarks; never exceed $totalMarks>,
  "confidencePercent": <float 0-100>,
  "goodPoints": "<bullet-point list — each point should sound like a real teacher praising the student>",
  "improvements": "<bullet-point list — each point should sound like a real teacher pointing out what to fix and why>",
  "finalReview": "<2-3 sentence handwritten-note-style overall review — warm, personal, constructive>",
  "annotations": [
    {
      "pageIndex": <int 0-indexed. Maximum is ${pageCount - 1}>,
      "yPositionPercent": <float 0-100>,
      "xStartPercent": <float 0-100>,
      "xEndPercent": <float 0-100>,
      "comment": "<short, warm teacher remark>",
      "isPositive": true,
      "lineStyle": "straight"
    }
  ]
}
NOTE: Every annotation MUST have "isPositive": true. Do NOT produce negative/cross annotations. Mention errors only in the "improvements" field.
''';

    final body = {
      'contents': [
        {
          'parts': [
            {'text': prompt},
          ],
        },
      ],
      'generationConfig': {
        'temperature': 0.2,
        'maxOutputTokens': 8192,
        'responseMimeType': 'application/json',
      },
    };

    dev.log(
      '[GeminiService] CachedOCR re-analysis → $pageCount page(s), Marks: $totalMarks',
      name: 'GeminiService',
    );

    // Retry loop: up to 3 attempts for a clean JSON response.
    const maxAttempts = 3;
    String? lastCleaned;
    Map<String, dynamic>? data;
    var usageAcc = GeminiTokenUsage.zero;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      final gen = await _callGeminiRaw(body, label: 'CachedOCR');
      lastCleaned = gen.text;
      usageAcc = usageAcc + (gen.tokenUsage ?? GeminiTokenUsage.zero);
      try {
        data = jsonDecode(gen.text) as Map<String, dynamic>;
        if (attempt > 1) {
          dev.log(
            '[GeminiService] CachedOCR JSON OK on retry $attempt.',
            name: 'GeminiService',
          );
        }
        break;
      } catch (e) {
        dev.log(
          '[GeminiService] CachedOCR JSON parse failed (attempt $attempt/$maxAttempts): $e',
          name: 'GeminiService',
        );
        if (attempt < maxAttempts) {
          await Future.delayed(Duration(milliseconds: 300 * attempt));
        }
      }
    }
    data ??= _parseAnalysisJson(lastCleaned ?? '{}');

    final rawAnnotations = data['annotations'] as List? ?? [];
    // pageCount is the number of pages sent for this block.
    // Clamp every annotation's pageIndex to [0, pageCount-1] so that Gemini
    // returning an out-of-range index (e.g. pageIndex: 1 on a 1-page question)
    // never causes annotations to be silently filtered out in the UI.
    final maxPageIdx = (pageCount - 1).clamp(0, pageCount);
    final annotations = rawAnnotations.whereType<Map>().map((a) {
      final map = a.cast<String, dynamic>();
      final rawPageIdx = (map['pageIndex'] as num?)?.toInt() ?? 0;
      return TeacherAnnotation(
        pageIndex: rawPageIdx.clamp(0, maxPageIdx),
        yPositionPercent: (map['yPositionPercent'] as num?)?.toDouble() ?? 50.0,
        xStartPercent: (map['xStartPercent'] as num?)?.toDouble() ?? 20.0,
        xEndPercent: (map['xEndPercent'] as num?)?.toDouble() ?? 80.0,
        comment: map['comment'] as String? ?? '',
        isPositive: map['isPositive'] as bool? ?? false,
        lineStyle: map['lineStyle'] as String? ?? 'straight',
      );
    }).toList();

    String parseStringOrList(dynamic val) {
      if (val is List) return val.map((e) => '• $e').join('\n');
      return val?.toString() ?? '';
    }

    final rawMarks = (data['marksAwarded'] as num?)?.toDouble() ?? 0.0;
    final marksAwarded = rawMarks.clamp(0.0, totalMarks.toDouble());
    final rawConfidence =
        (data['confidencePercent'] as num?)?.toDouble() ?? 0.0;

    return GeminiAnalysisOutput(
      studentText: cachedStudentText,
      marksAwarded: marksAwarded,
      confidencePercent: rawConfidence.clamp(0.0, 100.0),
      goodPoints: parseStringOrList(data['goodPoints']),
      improvements: parseStringOrList(data['improvements']),
      finalReview: data['finalReview'] as String? ?? '',
      annotations: annotations,
      tokenUsage: usageAcc.hasAnyCount ? usageAcc : null,
    );
  }

  static Future<GeminiAnalysisOutput> analyse({
    required List<Uint8List> pageImages,
    required String questionTitle,
    String? instructionName,
    required String modelDescription,
    required int totalMarks,
    String language = 'en',
    String checkLevel = 'Moderate',
  }) async {
    final checkLevelInstruction = checkLevel.toLowerCase() == 'hard'
        ? '- EVALUATION STRICTNESS: HARD. Be extremely strict. All answers must be strictly evaluated and normally score less than 50% of the total marks unless they are absolutely perfect without any flaws.'
        : '- EVALUATION STRICTNESS: MODERATE. Grade normally, but keep medium or average answers around or below 50% of the total marks.';

    final prompt =
        '''
You are a warm, experienced school teacher who genuinely cares about students improving. You are grading a handwritten student answer sheet. You write feedback the same way a real teacher would — personal, specific, encouraging where deserved, and honest where correction is needed.

You will be provided with images of the student's answer pages in sequence. Some pages may appear mostly white or contain very little visible ink — the student may have written lightly or used a light pen. ALWAYS attempt to read all pages. If a page genuinely has no answer at all, note it but still grade the rest of the answer.

${instructionName != null && instructionName.isNotEmpty ? 'EXTRA ANSWER INSTRUCTIONS:\n$instructionName\n\n' : ''}QUESTION TITLE:
$questionTitle

MODEL ANSWER / MARKING SCHEME (what the teacher expects):
$modelDescription

TOTAL MARKS FOR THIS QUESTION: $totalMarks

MARKING RULES (NON-NEGOTIABLE — apply like a strict board examiner):
$checkLevelInstruction
- Award marksAwarded as a DECIMAL in multiples of 0.5 (e.g. 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 …). NEVER exceed $totalMarks.
- Marking tiers by question size:
    8-mark question  → Bahot achha (excellent): 3.5  |  Moderate: 2.5  |  Low: 1.5
    12-mark question → Bahot achha (excellent): 5    |  Moderate: 3    |  Low: 2.5
    Other totals     → scale proportionally using 0.5-step decimals.
- You are a STRICT examiner. Default to FEWER marks, not more.
- FULL marks ONLY if answer nearly perfectly matches the scheme — all key points, correct terminology, clear reasoning. This is rare.
- Do NOT reward effort or length — reward accuracy and relevance only.
- A long but mostly irrelevant answer scores LOW. A short precise answer can outscore a long vague one.
- Do NOT give benefit of the doubt. If a key point is not clearly stated, do not assume it was implied.
- Deduct for: wrong facts, missing key terms, no examples when required, incorrect conclusions.
- When in doubt between two values, always choose the LOWER one.

YOUR TASKS:

1. READ the handwritten text from all images (including labels, captions, diagram text).
2. GRADE objectively against the marking scheme. If the scheme expects diagrams/figures, evaluate whether the student addressed those (sketches, descriptions, labels).
3. ANNOTATE the answer: mark 2–5 specific spots in the student's writing with short teacher-style comments. Make annotations PRECISE — annotate only the specific word/phrase, not the whole line.

ANNOTATION PLACEMENT RULES (CRITICAL — prevents overlapping comments):
- On any single page, every annotation MUST have a yPositionPercent that differs by AT LEAST 12 from every other annotation on that same page. Space them out evenly across the page height.
- If two annotations are naturally close together (within 12% of each other vertically), pick only the more important one; do NOT place both at nearly the same y position.
- Alternate comment placement: for annotations that are on the LEFT half of the text (xEndPercent < 55), keep xEndPercent ≤ 55. For annotations on the RIGHT half (xStartPercent > 45), keep xStartPercent ≥ 45. This ensures comments are anchored to different horizontal zones and do not collide.
- Prefer spreading annotations across different pages when the answer spans multiple pages — do not cluster all annotations on page 0.
- IMPORTANT: ALL annotations MUST have "isPositive": true. Only circle/annotate things done correctly. Do NOT mark errors with annotations — errors should be mentioned in the "improvements" field instead.

TONE & LANGUAGE GUIDELINES:

Write ALL feedback as a professional yet approachable teacher would — clear, constructive, and specific. Avoid robotic phrasing. Think of the tone an experienced senior examiner uses: direct, helpful, respectful.

GOOD annotation comments (professional but human):
  - "Well articulated — this directly addresses the marking scheme."
  - "Correct definition. Clear and concise."
  - "This definition is inaccurate — please revise from your textbook."
  - "You started well but left this point incomplete. Elaborate further next time."
  - "Good reasoning, but the correct formula is F = ma, not F = mv."
  - "Diagram is present but labels are missing — always label axes and units."

BAD (too robotic/generic — AVOID these):
  - "The student correctly identified the concept."
  - "Improvement needed in this area."
  - "Good point."
  - "Incorrect."

For goodPoints: Address the student directly and be specific about what was done well.
For improvements: Be specific — mention what was missed and why it matters for marks.
For finalReview: Write 2–3 sentences as a professional remark — constructive and encouraging.

${language == 'hi' ? '''LANGUAGE INSTRUCTION:
This is a Hindi-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in Hindi. No English feedback text at all.

HINDI VOCABULARY RULES (apply strictly):
- Use "saraahaniyan" or "Utkrisht" for praise — NOT just "achha".
- Use "Prayas" when noting effort.
- Nishkarsh (conclusion) must be future-oriented: "Aapka nishkarsh bhavishya unmukhi hona chahiye."
- Address the student as "Aap" / "Aapka" — NEVER "Tumne" or "Tu".
- "Introduction" → write "Prashtavana" or "Parichay" — never the English word.
- NEVER use the word "Shabash".
- "Utpatti" → use "Prashtuti".
- Strong conclusion: "Aapka nishkarsh prabhavshali hai."
- Line could be more specific: "Yah line aur vishishth ho sakti hai; udaharan ke sath samjhaya ja sakta tha."
- Replace "Sahi dhang se samjhaya hai" with "Sahi dhang se prastut kiya hai."''' : '''LANGUAGE INSTRUCTION:
This is an English-medium question paper. Write ALL feedback fields (goodPoints, improvements, finalReview, annotation comments) ENTIRELY in English. No Hindi feedback text at all.'''}

══════════════════════════════════════════
OUTPUT FORMAT:
══════════════════════════════════════════
Return ONLY a valid JSON object (no markdown, no explanation outside JSON) with exactly these keys:
{
  "studentText": "<transcribe the exact handwritten text you see across all pages — skip printed question headers>",
  "marksAwarded": <decimal in multiples of 0.5, range 0 to $totalMarks; never exceed $totalMarks>,
  "confidencePercent": <float 0-100>,
  "goodPoints": "<bullet-point list — each point should sound like a real teacher praising the student>",
  "improvements": "<bullet-point list — each point should sound like a real teacher pointing out what to fix and why>",
  "finalReview": "<2-3 sentence handwritten-note-style overall review — warm, personal, constructive>",
  "annotations": [
    {
      "pageIndex": <int 0-indexed corresponding to the image sequence. Maximum is ${pageImages.length - 1}>,
      "yPositionPercent": <float 0-100 indicating the approximate vertical position of the specific text to underline>,
      "xStartPercent": <float 0-100 indicating the tight horizontal start of the specific word(s) to underline>,
      "xEndPercent": <float 0-100 indicating the tight horizontal end of the specific word(s) to underline>,
      "comment": "<short, warm, colloquial teacher remark — praise what's right, sound human>",
      "isPositive": true,
      "lineStyle": "straight"
    }
  ]
}
NOTE: Every annotation MUST have "isPositive": true. Do NOT produce any negative/cross annotations. Errors should be mentioned only in the "improvements" field.
''';

    final parts = <Map<String, dynamic>>[
      {'text': prompt},
    ];

    // Send all pages — Gemini is smart enough to identify where writing exists.
    for (final pageBytes in pageImages) {
      parts.add({
        'inline_data': {
          'mime_type': 'image/jpeg',
          'data': base64Encode(pageBytes),
        },
      });
    }

    final body = {
      'contents': [
        {'parts': parts},
      ],
      'generationConfig': {
        'temperature': 0.2, // low temp for objective grading
        'maxOutputTokens': 8192,
        'responseMimeType': 'application/json',
      },
    };

    // Retry loop: try up to 3 times to get a clean, parseable JSON response.
    // Only fall back to repair/regex after all attempts are exhausted.
    // Token logs use: "Combined Analysis → N page(s), Marks: M · FullAnalysis → …"
    final fullAnalysisLabel =
        'Combined Analysis → ${pageImages.length} page(s), Marks: $totalMarks · FullAnalysis';
    const maxAttempts = 3;
    String? lastCleaned;
    Map<String, dynamic>? data;
    var usageAcc = GeminiTokenUsage.zero;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      final gen = await _callGeminiRaw(body, label: fullAnalysisLabel);
      lastCleaned = gen.text;
      usageAcc = usageAcc + (gen.tokenUsage ?? GeminiTokenUsage.zero);
      try {
        data = jsonDecode(gen.text) as Map<String, dynamic>;
        if (attempt > 1) {
          dev.log(
            '[GeminiService] Analysis JSON OK on retry $attempt.',
            name: 'GeminiService',
          );
        }
        break; // clean parse — no need to retry
      } catch (e) {
        dev.log(
          '[GeminiService] Analysis JSON parse failed (attempt $attempt/$maxAttempts): $e',
          name: 'GeminiService',
        );
        if (attempt < maxAttempts) {
          await Future.delayed(Duration(milliseconds: 300 * attempt));
        }
      }
    }
    // All retries exhausted — use repair / regex fallback on the last response.
    data ??= _parseAnalysisJson(lastCleaned ?? '{}');

    final rawAnnotations = data['annotations'] as List? ?? [];
    // Clamp every annotation's pageIndex to [0, pageImages.length-1] so that
    // Gemini returning an out-of-range index never silently hides annotations.
    final maxPageIdx = (pageImages.length - 1).clamp(0, pageImages.length);
    final annotations = rawAnnotations.whereType<Map>().map((a) {
      final map = a.cast<String, dynamic>();
      final rawPageIdx = (map['pageIndex'] as num?)?.toInt() ?? 0;
      return TeacherAnnotation(
        pageIndex: rawPageIdx.clamp(0, maxPageIdx),
        yPositionPercent: (map['yPositionPercent'] as num?)?.toDouble() ?? 50.0,
        xStartPercent: (map['xStartPercent'] as num?)?.toDouble() ?? 20.0,
        xEndPercent: (map['xEndPercent'] as num?)?.toDouble() ?? 80.0,
        comment: map['comment'] as String? ?? '',
        isPositive: map['isPositive'] as bool? ?? false,
        lineStyle: map['lineStyle'] as String? ?? 'straight',
      );
    }).toList();

    String parseStringOrList(dynamic val) {
      if (val is List) {
        return val.map((e) => '• $e').join('\n');
      }
      return val?.toString() ?? '';
    }

    // Strict marks validation: clamp to [0, totalMarks]
    final rawMarks = (data['marksAwarded'] as num?)?.toDouble() ?? 0.0;
    final marksAwarded = rawMarks.clamp(0.0, totalMarks.toDouble());
    if (rawMarks != marksAwarded) {
      dev.log(
        '[GeminiService] marksAwarded $rawMarks clamped to valid range [0, $totalMarks] → $marksAwarded',
        name: 'GeminiService',
      );
    }

    final rawConfidence =
        (data['confidencePercent'] as num?)?.toDouble() ?? 0.0;
    final confidencePercent = rawConfidence.clamp(0.0, 100.0);

    return GeminiAnalysisOutput(
      studentText: data['studentText'] as String? ?? '',
      marksAwarded: marksAwarded,
      confidencePercent: confidencePercent,
      goodPoints: parseStringOrList(data['goodPoints']),
      improvements: parseStringOrList(data['improvements']),
      finalReview: data['finalReview'] as String? ?? '',
      annotations: annotations,
      tokenUsage: usageAcc.hasAnyCount ? usageAcc : null,
    );
  }
}

/// Top-level function necessary for compute() to run in an isolate
Future<GeminiAnalysisOutput> _analyseIsolateHandler(
  Map<String, dynamic> args,
) async {
  return GeminiService.analyse(
    pageImages: args['pageImages'] as List<Uint8List>,
    questionTitle: args['questionTitle'] as String,
    instructionName: args['instructionName'] as String?,
    modelDescription: args['modelDescription'] as String,
    totalMarks: args['totalMarks'] as int,
    language: args['language'] as String? ?? 'en',
    checkLevel: args['checkLevel'] as String? ?? 'Moderate',
  );
}

/// Top-level handler for cached-OCR re-analysis in isolate.
Future<GeminiAnalysisOutput> _analyseWithCachedOcrHandler(
  Map<String, dynamic> args,
) async {
  return GeminiService._analyseFromCachedText(
    cachedStudentText: args['cachedStudentText'] as String,
    questionTitle: args['questionTitle'] as String,
    instructionName: args['instructionName'] as String?,
    modelDescription: args['modelDescription'] as String,
    totalMarks: args['totalMarks'] as int,
    pageCount: args['pageCount'] as int,
    language: args['language'] as String? ?? 'en',
    checkLevel: args['checkLevel'] as String? ?? 'Moderate',
  );
}

/// Top-level handler for combined key review in isolate.
Future<CombinedKeyReviewOutput> _combinedKeyReviewHandler(
  Map<String, dynamic> args,
) async {
  return GeminiService._doCombinedKeyReview(
    questionResults: (args['questionResults'] as List)
        .cast<Map<String, dynamic>>(),
  );
}

/// Top-level handler for intro-page analysis in isolate.
Future<IntroPageAnalysis> _analyseIntroPageHandler(
  Map<String, dynamic> args,
) async {
  return GeminiService._doAnalyseIntroPage(args['pageImage'] as Uint8List);
}
