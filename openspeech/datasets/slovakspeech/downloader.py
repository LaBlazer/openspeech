import urllib, sys, time


def _chunk_read(response, file_path, chunk_size=1024 * 128):
    total_size_mbytes = response.info().get('Content-Length').strip()
    total_size_mbytes = float(total_size_mbytes) / 1000000.0
    mbytes_so_far = 0.0
    start = time.time()

    with open(file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            mbytes_so_far += len(chunk) / 1000000.0

            if not chunk:
                break

            f.write(chunk)

            percent = mbytes_so_far / total_size_mbytes * 100.0
            sys.stdout.write("Downloaded %d of %d MB (%0.2f%%, %0.2f MBps)\r" % 
                (mbytes_so_far, total_size_mbytes, percent, mbytes_so_far/(time.time() - start)))

            if mbytes_so_far >= total_size_mbytes:
                sys.stdout.write('\n')

def download(url, output_path):
   req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            })
   response = urllib.request.urlopen(req)
   _chunk_read(response, output_path)