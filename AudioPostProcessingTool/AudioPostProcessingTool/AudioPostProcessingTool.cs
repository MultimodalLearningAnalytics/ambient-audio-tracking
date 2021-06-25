using System;
using Microsoft.Psi;
using Microsoft.Psi.Audio;
using Microsoft.Psi.Data;
using Microsoft.Psi.Interop.Format;
using Microsoft.Psi.Interop.Transport;

namespace AudioPostProcessingTool
{
    class AudioPostProcessingTool
    {
        static void Main(string[] args)
        {
            using (var pipeline = Pipeline.Create()) {
                while (true)
                {
                    Console.WriteLine("Please choose a processing option: \r\n\t1) Write stream to wave file\r\n\t2) Test timestamp formating\r\n\t3) Reprocess peak frequency (apply under compression)\r\n\t4) Pipe audio to python and ingest distraction prediction\r\n\t5) Ingest a csv with prediction\r\n\t6) Apply an additional filter to the prediction stores\r\n\t7) Process audio and phone data to multimodal predictions\r\n\t8) simplify prediction stream to 0 or 1");
                    string input = Console.ReadLine();

                    if (input == "1")
                    {
                        Console.WriteLine("Please provide the path to the dataset");
                        string inpath = Console.ReadLine();
                        Console.WriteLine("Please provide a file name for the output file");
                        string fn = Console.ReadLine();
                        Console.WriteLine("Starting script!");

                        WriteAudio(pipeline, inpath, fn);
                    } else if (input == "2") {
                        DateTime ts = DateTime.Now;
                        Console.WriteLine(ts.ToString("yy_MM_dd_HH_mm_ss"));
                    } else if (input == "3") {
                        Console.WriteLine("Please provide the path to the dataset");
                        string inpath = Console.ReadLine();
                        ReprocessFrequency(pipeline, inpath);
                    } else if (input == "4") {
                        Console.WriteLine("Please provide the path to the dataset");
                        string inpath = Console.ReadLine();
                        PipeToPython(pipeline, inpath);
                    } else if (input == "5") {
                        Console.WriteLine("Please provide the path to the dataset");
                        string storepath = Console.ReadLine();
                        Console.WriteLine("Please provide the path to the regular csv file");
                        string csvpath = Console.ReadLine();
                        Console.WriteLine("Please provide the path to the filtered csv file");
                        string csvpathFilter = Console.ReadLine();
                        Console.WriteLine("Please provide the name for the output stream");
                        string name = Console.ReadLine();
                        Console.WriteLine("Please provide the start time");
                        string startTime = Console.ReadLine();
                        PredictionCsvIngest(pipeline, csvpath, csvpathFilter, storepath, name, startTime);
                    } else if (input == "6") {
                        Console.WriteLine("Please provide the path to the dataset");
                        string inpath = Console.ReadLine();
                        ReprocessPredictions(pipeline, inpath);
                    } else if (input == "7") {
                        Console.WriteLine("Please provide the path to the audio dataset");
                        string audiopath = Console.ReadLine();
                        Console.WriteLine("Please provide the path to the phone dataset");
                        string phonepath = Console.ReadLine();
                        Console.WriteLine("Please provide the path where to store the new ds");
                        string outputpath = Console.ReadLine();
                        MultiModalMerge(pipeline, audiopath, phonepath, outputpath);
                    } else if (input == "8") {
                        Console.WriteLine("Please provide the path to the input dataset");
                        string inpath = Console.ReadLine();
                        Console.WriteLine("Please provide the name of the input partition");
                        string inname = Console.ReadLine();
                        Console.WriteLine("Please provide the name of the input stream");
                        string instreamname = Console.ReadLine();
                        Console.WriteLine("Please provide the path where to store the output ds");
                        string outpath = Console.ReadLine();
                        Console.WriteLine("Please provide the name of the output stream and partition");
                        string outname = Console.ReadLine();
                        Console.WriteLine("Please provide the start of the ds");
                        string start = Console.ReadLine();

                        PredictionClassifier(pipeline, inpath, outpath, inname, instreamname, outname, start);
                    } else {
                        Console.WriteLine("unknown command");
                    }
                }
            }
        }

        private static void WriteAudio(Pipeline pipeline, string inpath, string fn) {
            Console.WriteLine("Loading store");
            var ds = Dataset.CreateFromStore(new PsiStoreStreamReader("AudioRawData", inpath), "TempPartition");
            /*var store = PsiStore.Open(pipeline, "AudioRawData", inpath);*/

            // create an object for reporting progress
            var progress = new Progress<(string, double)>(p => Console.WriteLine($"[{p.Item2:P1}] {p.Item1}"));

            ds.CreateDerivedPartitionAsync(
                (pipeline, importer, exporter) =>
                {
                    DateTime ts = DateTime.Now;

                    System.IO.Directory.CreateDirectory($"E:\\wav_files\\{fn}\\");

                    var rawAudioLav = importer.OpenStream<AudioBuffer>("LavAudioRaw");
                    var audioWriterLav = new WaveFileWriter(pipeline, $"E:\\wav_files\\{fn}\\{ts.ToString("yy_MM_dd_HH_mm_ss")}_{fn}_lav.wav");
                    rawAudioLav.PipeTo(audioWriterLav);

                    var rawAudioEp = importer.OpenStream<AudioBuffer>("EpAudioRaw");
                    var audioWriterEp = new WaveFileWriter(pipeline, $"E:\\wav_files\\{fn}\\{ts.ToString("yy_MM_dd_HH_mm_ss")}_{fn}_ep.wav");
                    rawAudioEp.PipeTo(audioWriterEp);

                },
                "TROEP",
                false,
                "TROEP",
                "E\\ZOOI\\",
                null,
                null,
                false,
                progress: progress).Wait();

            /*DateTime ts = DateTime.Now;

            var rawAudioLav = store.OpenStream<AudioBuffer>("LavAudioRaw");
            var audioWriterLav = new WaveFileWriter(pipeline, $"E:\\wav_files\\{ts}_{fn}_lav.wav");
            rawAudioLav.PipeTo(audioWriterLav);

            var rawAudioEp = store.OpenStream<AudioBuffer>("EpAudioRaw");
            var audioWriterEp = new WaveFileWriter(pipeline, $"E:\\wav_files\\{ts}_{fn}_ep.wav");
            rawAudioEp.PipeTo(audioWriterEp);*/

        }

        private static void ReprocessFrequency(Pipeline pipeline, string inpath)
        {
            Console.WriteLine("Loading store");
            var ds = Dataset.CreateFromStore(new PsiStoreStreamReader("AudioRawData", inpath), "TempPartition");
            /*var store = PsiStore.Open(pipeline, "AudioRawData", inpath);*/

            // create an object for reporting progress
            var progress = new Progress<(string, double)>(p => Console.WriteLine($"[{p.Item2:P1}] {p.Item1}"));

            ds.CreateDerivedPartitionAsync(
                (pipeline, importer, exporter) =>
                {
                    DateTime ts = DateTime.Now;

                    PsiExporter store = PsiStore.Create(pipeline, "NewAudioFreq", inpath);

                    var lastLav = 1.0;
                    var secondLav = 1.0;

                    var lavFreq = importer.OpenStream<double>("LavAudioFreq").Where(m => {

                        if (m == 1 && (lastLav != 1 || secondLav != 1)) {
                            secondLav = lastLav;
                            lastLav = m;

                            return false;
                        }

                        secondLav = lastLav;
                        lastLav = m;
                        return m >= 100 || m < 10;
                    
                    });
                    lavFreq.Write("NewLavAudioFreq", store);

                    var lastEp = 1.0;
                    var secondEp = 1.0;

                    var epFreq = importer.OpenStream<double>("LavAudioFreq").Where(m => {

                        if (m == 1 && (lastEp != 1 || secondEp != 1))
                        {
                            secondEp = lastEp;
                            lastEp = m;
                            return false;
                        }

                        secondEp = lastEp;
                        lastEp = m;
                        return m >= 100 || m < 10;

                    });
                    epFreq.Write("newEpAudioFreq", store);

                },
                "TROEP",
                false,
                "TROEP",
                "E\\ZOOI\\",
                null,
                null,
                false,
                progress: progress).Wait();

            /*DateTime ts = DateTime.Now;

            var rawAudioLav = store.OpenStream<AudioBuffer>("LavAudioRaw");
            var audioWriterLav = new WaveFileWriter(pipeline, $"E:\\wav_files\\{ts}_{fn}_lav.wav");
            rawAudioLav.PipeTo(audioWriterLav);

            var rawAudioEp = store.OpenStream<AudioBuffer>("EpAudioRaw");
            var audioWriterEp = new WaveFileWriter(pipeline, $"E:\\wav_files\\{ts}_{fn}_ep.wav");
            rawAudioEp.PipeTo(audioWriterEp);*/

        }

        private static void PipeToPython(Pipeline pipeline, string inpath) {
            Console.WriteLine("Loading store");
            var ds = Dataset.CreateFromStore(new PsiStoreStreamReader("AudioRawData", inpath), "TempPartition");
            /*var store = PsiStore.Open(pipeline, "AudioRawData", inpath);*/

            // create an object for reporting progress
            var progress = new Progress<(string, double)>(p => Console.WriteLine($"[{p.Item2:P1}] {p.Item1}"));

            ds.CreateDerivedPartitionAsync(
                (pipeline, importer, exporter) =>
                {
                    DateTime ts = DateTime.Now;

                    PsiExporter store = PsiStore.Create(pipeline, "AudioPredictions", inpath);

                    var lavAudio = importer.OpenStream<AudioBuffer>("LavAudioRaw");
                    var mq = new NetMQWriter<AudioBuffer>(pipeline, "AudioRaw", "tcp://localhost:12345", JsonFormat.Instance);
                    lavAudio.PipeTo(mq);

                    var mqi = new NetMQSource<double>(pipeline, "LavAudioPred", "tcp://localhost:12346", JsonFormat.Instance);
                    mqi.Do(x => Console.WriteLine($"Message: {x}"));
                    /*mqi.Write("LavAudioPrediction", store);*/

                },
                "TROEP",
                false,
                "TROEP",
                "E\\ZOOI\\",
                null,
                null,
                false,
                progress: progress).Wait();
        }

        private static void PredictionCsvIngest(Pipeline pipeline, string csvpath, string csvpathFilter, string storepath, string name, string startTime) {
            DateTime startDate = DateTime.Parse(startTime);
            PsiExporter store = PsiStore.Create(pipeline, "AudioPredictions", storepath);

            AudioPredictionGenerator gen = new AudioPredictionGenerator(pipeline, csvpath, startDate, name);
            AudioPredictionGenerator genFilter = new AudioPredictionGenerator(pipeline, csvpathFilter, startDate, name + "-filtered");
            gen.Pred.Write(name, store);
            genFilter.Pred.Write(name + "-filtered", store);

            pipeline.Run();

        }

        private static void ReprocessPredictions(Pipeline pipeline, string inpath)
        {
            Console.WriteLine("Loading store");
            var ds = Dataset.CreateFromStore(new PsiStoreStreamReader("AudioPredictions", inpath), "TempPartition");
            /*var store = PsiStore.Open(pipeline, "AudioRawData", inpath);*/

            // create an object for reporting progress
            var progress = new Progress<(string, double)>(p => Console.WriteLine($"[{p.Item2:P1}] {p.Item1}"));

            ds.CreateDerivedPartitionAsync(
                (pipeline, importer, exporter) =>
                {
                    DateTime ts = DateTime.Now;

                    PsiExporter store = PsiStore.Create(pipeline, "AudioPrediction-filtered", inpath);

                    var last = 0.0;
                    var second = 0.0;

                    var unfiltered = importer.OpenStream<double>("AudioPrediction").Where(m => {

                        if (m == 0 && (last != 0 || second != 0))
                        {
                            second = last;
                            last = m;

                            return false;
                        }

                        second = last;
                        last = m;
                        return true;

                    });
                    unfiltered.Write("AudioPrediction", store);

                    var lastF = 0.0;
                    var secondF = 0.0;

                    var filtered = importer.OpenStream<double>("AudioPrediction-filtered").Where(m => {

                        if (m == 0 && (lastF != 0 || secondF != 0))
                        {
                            secondF = lastF;
                            lastF = m;
                            return false;
                        }

                        secondF = lastF;
                        lastF = m;
                        return m >= 100 || m < 10;

                    });
                    filtered.Write("AudioPrediction-filtered", store);

                },
                "TROEP",
                false,
                "TROEP",
                "E\\ZOOI\\",
                null,
                null,
                false,
                progress: progress).Wait();

            /*DateTime ts = DateTime.Now;

            var rawAudioLav = store.OpenStream<AudioBuffer>("LavAudioRaw");
            var audioWriterLav = new WaveFileWriter(pipeline, $"E:\\wav_files\\{ts}_{fn}_lav.wav");
            rawAudioLav.PipeTo(audioWriterLav);

            var rawAudioEp = store.OpenStream<AudioBuffer>("EpAudioRaw");
            var audioWriterEp = new WaveFileWriter(pipeline, $"E:\\wav_files\\{ts}_{fn}_ep.wav");
            rawAudioEp.PipeTo(audioWriterEp);*/

        }

        private static void MultiModalMerge(Pipeline pipeline, string audiopath, string phonepath, string outputpath)
        {
            Console.WriteLine("Loading store");
            var audiostore = PsiStore.Open(pipeline, "AudioPrediction-filtered", audiopath);
            var phonestore = PsiStore.Open(pipeline, "PhonePrediction", phonepath);
            var outputstore = PsiStore.Create(pipeline, "MultimodalPrediction", outputpath);

            var streamAudio = audiostore.OpenStream<Double>("AudioPrediction");
            var streamPhone = phonestore.OpenStream<Double>("prediction");

            var joined = streamPhone.Join(streamAudio, Reproducible.Nearest<Double>(RelativeTimeInterval.Future()));
            /*joined.Out.Process<(double, double), double>(
                (x, e, o) =>
                {
                    var p = x.Item1;
                    var a = x.Item2;

                    TimeSpan offset = new TimeSpan((e.OriginatingTime.Ticks - phoneStore.StartTime.Ticks));
                    contentPhoneAccelerometer += e.OriginatingTime.ToString("yyyy/MM/dd HH:mm:ss.fff") + ", " + offset.TotalSeconds + ", " + x.Item1 + ", " + x.Item2 + ", " + x.Item3 + ", 1.0\n";
                });*/

            IProducer<double> output = joined.Select(((double, double) x) => {
                double p = x.Item1;
                double a = x.Item2;

                Console.WriteLine($"Processing: {x}");

                if (p >= 0.5) {
                    return p;
                /*} else if (p >= 0.5 && a >= 0.5) {
                    return ((98.57 * p) + (62.07 * a)) / (98.57 + 62.07);
                } else if (p >= 0.5 && a < 0.5) {
                    return ((98.57 * p) + (59.26 * a)) / (98.57 + 59.62);*/
                } else if (p < 0.5 && a >= 0.5) {
                    return ((87.78 * p) + (62.07 * a)) / (87.78 + 62.07);
                } else {
                    return ((87.78 * p) + (59.26 * a)) / (87.78 + 59.26);
                }
            });

            output.Write("MultimodalPrediction", outputstore);

            pipeline.Run(new ReplayDescriptor(DateTime.Parse("2021-06-01T14:54:00.3514000+02:00"), false));

            Console.WriteLine("FINITO");

        }

        private static void PredictionClassifier(Pipeline pipeline, string inpath, string outpath, string inname, string instreamname, string outname, string start) {
            Console.WriteLine("Loading stores");
            var instore = PsiStore.Open(pipeline, inname, inpath);
            var outputstore = PsiStore.Create(pipeline, outname, outpath);
            Console.WriteLine("Stores loaded");

            IProducer<double> streamAudio = instore.OpenStream<Double>(instreamname).Select(m => {
                Console.WriteLine($"Processing: {m}");
                if (m >= 0.5) {
                    return 1.0;
                } else {
                    return 0.0;
                }
            });

            streamAudio.Write(outname, outputstore);
            pipeline.Run(new ReplayDescriptor(DateTime.Parse(start), false));

            Console.WriteLine("Done");
        }

    }
}
