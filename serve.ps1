param(
    [int]$Port = 8000,
    [string]$Root = $PSScriptRoot
)

Add-Type -AssemblyName System.Net.HttpListener -ErrorAction SilentlyContinue

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$Port/")
$listener.Start()
Write-Host "Serving $Root on http://localhost:$Port/"

$mime = @{
    ".html" = "text/html; charset=utf-8"
    ".js"   = "text/javascript; charset=utf-8"
    ".css"  = "text/css; charset=utf-8"
    ".json" = "application/json; charset=utf-8"
    ".png"  = "image/png"
    ".jpg"  = "image/jpeg"
    ".jpeg" = "image/jpeg"
}

try {
    while ($listener.IsListening) {
        $ctx = $listener.GetContext()
        $req = $ctx.Request
        $res = $ctx.Response
        try {
            $path = $req.Url.LocalPath
            if ($path -eq "/") { $path = "/index.html" }
            $filePath = Join-Path $Root ($path.TrimStart("/"))
            $filePath = [System.IO.Path]::GetFullPath($filePath)
            if (-not $filePath.StartsWith([System.IO.Path]::GetFullPath($Root))) {
                $res.StatusCode = 403
                $res.Close()
                continue
            }
            if (Test-Path $filePath -PathType Leaf) {
                $ext = [System.IO.Path]::GetExtension($filePath)
                $contentType = $mime[$ext]
                if (-not $contentType) { $contentType = "application/octet-stream" }
                $res.ContentType = $contentType
                $res.Headers.Add("Cache-Control", "no-store, no-cache, must-revalidate")
                $bytes = [System.IO.File]::ReadAllBytes($filePath)
                $res.ContentLength64 = $bytes.Length
                $res.OutputStream.Write($bytes, 0, $bytes.Length)
            } else {
                $res.StatusCode = 404
                $msg = [System.Text.Encoding]::UTF8.GetBytes("Not found: $path")
                $res.OutputStream.Write($msg, 0, $msg.Length)
            }
        } catch {
            $res.StatusCode = 500
        } finally {
            $res.Close()
        }
    }
} finally {
    $listener.Stop()
}
