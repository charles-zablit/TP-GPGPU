#include "log.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>

void log_debug(const char *message, ...)
{
#ifdef DEBUG
   static clock_t start_time = clock();
   printf("[DEBUG %f] ", (double)(clock() - start_time) / CLOCKS_PER_SEC);
   va_list args;
   va_start(args, message);
   vprintf(message, args);
   va_end(args);
   printf("\n");
#endif
}

void log_info(const char *message, ...)
{
   static clock_t start_time = clock();
   printf("[INFO %f] ", (double)(clock() - start_time) / CLOCKS_PER_SEC);
   va_list args;
   va_start(args, message);
   vprintf(message, args);
   va_end(args);
   printf("\n");
}