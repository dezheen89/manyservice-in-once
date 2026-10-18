<?php
/**
 * tts-proxy.php — server-side proxy for the CosyVoice3 Badini RunPod endpoint.
 *
 * The browser never sees the RunPod API key. This file holds it privately
 * and forwards requests to RunPod.
 *
 * Frontend calls:
 *   POST /tts-proxy.php?action=run              body: {"text","speed","emotion","ref_audio","ref_text"}
 *   GET  /tts-proxy.php?action=status&id=JOBID
 *   GET  /tts-proxy.php?action=voices           returns the voice + available emotions
 *
 * SECURITY: only whitelisted fields are forwarded; the browser can never
 * trigger arbitrary payloads.
 *
 * CosyVoice3 is zero-shot voice cloning from bundled reference clips of one
 * speaker ("Hayfa Female") baked into the worker image — one clip per
 * emotion, selected by name. The model has no real emotion conditioning;
 * "emotion" just picks which real recording of her expressing that emotion
 * is used as the clone target. A caller may still override the reference
 * entirely via ref_audio/ref_text.
 */

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------

$ENDPOINT_ID = 'z0fg2nr3xccf3v';
$API_KEY     = 'YOUR_RUNPOD_API_KEY';

$ALLOWED_EMOTIONS = ['natural', 'angry', 'sad', 'happy', 'fear', 'surprise'];
$MAX_TEXT_LEN = 2500;
$MAX_REF_TEXT_LEN = 1000;
$MAX_BODY_BYTES   = 8 * 1024 * 1024;   // 8 MB — enough for a 30s reference clip
$ALLOWED_ORIGIN   = '*';

// ---------------------------------------------------------------------------

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: ' . $ALLOWED_ORIGIN);
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(204); exit; }

function fail($code, $msg) {
    http_response_code($code);
    echo json_encode(['error' => $msg]);
    exit;
}

if ($API_KEY === 'PUT_YOUR_RUNPOD_API_KEY_HERE' || $API_KEY === '') {
    fail(500, 'Proxy not configured: set RunPod API key in tts-proxy.php');
}

$action = isset($_GET['action']) ? $_GET['action'] : '';
$base   = "https://api.runpod.ai/v2/{$ENDPOINT_ID}";

if ($action === 'voices') {
    // The CosyVoice3 worker has no built-in voice-listing debug command
    // (unlike the old F5-TTS one) — it's a single speaker with a bundled
    // reference clip per emotion, so the proxy just reports that statically
    // instead of calling RunPod.
    echo json_encode([
        'voices' => [
            ['id' => 'hayfa-female', 'name' => 'Hayfa Female'],
        ],
        'emotions' => $ALLOWED_EMOTIONS,
    ]);
    exit;
}

if ($action === 'warmup') {
    // Tiny fire-and-forget job so a worker starts loading the model before a real request.
    $wch = curl_init("{$base}/run");
    curl_setopt($wch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($wch, CURLOPT_HTTPHEADER, ['Authorization: Bearer ' . $API_KEY, 'Content-Type: application/json']);
    curl_setopt($wch, CURLOPT_POST, true);
    curl_setopt($wch, CURLOPT_POSTFIELDS, json_encode(['input' => ['text' => 'سڵاو', 'speed' => 1.0, 'emotion' => 'natural']]));
    curl_setopt($wch, CURLOPT_TIMEOUT, 15);
    curl_exec($wch);
    curl_close($wch);
    echo json_encode(['ok' => true]);
    exit;
}

$ch = curl_init();
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Authorization: Bearer ' . $API_KEY,
    'Content-Type: application/json',
]);
curl_setopt($ch, CURLOPT_TIMEOUT, 60);

if ($action === 'run') {
    $raw = file_get_contents('php://input');
    if (strlen($raw) > $MAX_BODY_BYTES) {
        fail(413, 'Request too large');
    }
    $in = json_decode($raw, true);
    if (!is_array($in)) { fail(400, 'Invalid JSON body'); }

    $text    = isset($in['text'])    ? (string)$in['text']    : '';
    $speed   = isset($in['speed'])   ? (float)$in['speed']    : 1.0;
    $emotion = isset($in['emotion']) ? (string)$in['emotion'] : 'natural';

    if ($text === '') { fail(400, 'Missing text'); }
    if (mb_strlen($text) > $MAX_TEXT_LEN) { fail(400, "Text too long (max {$MAX_TEXT_LEN} chars)"); }
    if (!in_array($emotion, $ALLOWED_EMOTIONS, true)) { fail(400, 'Invalid emotion'); }
    if ($speed < 0.5) { $speed = 0.5; }
    if ($speed > 2.0) { $speed = 2.0; }

    $input = [
        'text'    => $text,
        'speed'   => $speed,
        'emotion' => $emotion,
    ];

    // Voice cloning override: caller-supplied reference audio + its exact
    // transcript, in place of the bundled per-emotion default.
    if (isset($in['ref_audio']) && $in['ref_audio'] !== '') {
        $refAudio = (string)$in['ref_audio'];
        $refText  = isset($in['ref_text']) ? (string)$in['ref_text'] : '';
        if ($refText === '') { fail(400, 'ref_text is required with ref_audio'); }
        if (mb_strlen($refText) > $MAX_REF_TEXT_LEN) { fail(400, 'ref_text too long'); }
        // base64 sanity check (character set only — cheap)
        if (!preg_match('/^[A-Za-z0-9+\/=\r\n]+$/', $refAudio)) {
            fail(400, 'ref_audio must be base64');
        }
        $input['ref_audio'] = $refAudio;
        $input['ref_text']  = $refText;
    }

    $payload = json_encode(['input' => $input]);

    curl_setopt($ch, CURLOPT_URL, "{$base}/run");
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);

} elseif ($action === 'status') {
    $id = isset($_GET['id']) ? $_GET['id'] : '';
    if ($id === '' || !preg_match('/^[A-Za-z0-9\-]+$/', $id)) {
        fail(400, 'Invalid or missing job id');
    }
    curl_setopt($ch, CURLOPT_URL, "{$base}/status/{$id}");
    curl_setopt($ch, CURLOPT_HTTPGET, true);

} else {
    fail(400, 'Unknown action (use ?action=run, ?action=status&id=..., or ?action=voices)');
}

$response = curl_exec($ch);
if ($response === false) {
    $err = curl_error($ch);
    curl_close($ch);
    fail(502, 'Upstream request failed: ' . $err);
}
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

http_response_code($httpCode ?: 200);
echo $response;
