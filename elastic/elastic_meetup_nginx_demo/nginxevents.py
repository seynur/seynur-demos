from random import choice, choices, randrange
from datetime import datetime, timedelta

# $remote_addr - $remote_user [$time_local]  "$request_method $request_uri HTTP/1.1" $status $body_bytes_sent "$http_referer" "$http_user_agent"
# 10.10.1.1 - - [01/Nov/2017:06:47:09 +0000] "GET /test.html HTTP/1.1" 404 8571 "-" "Mozilla/5.0 (compatible; Facebot 1.0; https://developers.facebook.com/docs/sharing/webmasters/crawler)"


if __name__ == '__main__':

    # TODO: Accept parameters and perform input validation/error checking (e.g. file not found etc.)
    # Read files and set variables here
    available_addr = list()
    with open('samples/external_ips.sample', 'rt') as f:
        for l in f:
            available_addr.append(l.strip())

    available_uris = list()
    with open('samples/uris.sample', 'rt') as f:
        for l in f:
            available_uris.append(l.strip())

    available_useragents = list()
    with open('samples/useragents.sample', 'rt') as f:
        for l in f:
            available_useragents.append(l.strip())

    # Set variables here - absolutely made up weights, no significance
    available_methods = choices(['GET','POST'],[38,7],k=30)
    available_statuses = choices(['200','302', '404', '500'],[38,7, 8, 3],k=100)

    # Choose date range to generate events
    # e.g. last 7 days --> days=7
    # e.g. last 3 hours --> hours=3
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    #for i in range(15):
    with open('logs/example_access.log', 'wt') as f:
        while start_date < end_date:
            remote_addr = choice(available_addr)
            remote_user = '-'
            time_local = start_date.strftime('%d/%b/%Y:%H:%M:%S +0000')
            request_method = choice(available_methods)
            request_uri = choice(available_uris)
            status = choice(available_statuses)
            body_bytes_sent = randrange(6789, 75032)
            http_referer = '-'
            http_user_agent = choice(available_useragents)

            line = "{} - {} [{}] \"{} {} HTTP/1.1\" {} {} \"{}\" \"{}\"\n"\
                .format(remote_addr, remote_user, time_local, request_method, request_uri, status, body_bytes_sent, http_referer, http_user_agent)

            f.write(line)
            #print(line)

            # increment date by random seconds
            start_date = start_date + timedelta(seconds=randrange(1, 10))
