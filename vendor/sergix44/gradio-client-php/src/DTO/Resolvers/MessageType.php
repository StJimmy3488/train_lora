<?php

namespace SergiX44\Gradio\DTO\Resolvers;

enum MessageType: string
{
    case SEND_HASH = 'send_hash';
    case SEND_DATA = 'send_data';
    case QUEUE_FULL = 'queue_full';
    case QUEUE_ESTIMATION = 'estimation';
    case PROCESS_STARTS = 'process_starts';
    case PROCESS_GENERATING = 'process_generating';
    case PROCESS_COMPLETED = 'process_completed';

    case LOG = 'log';
}
