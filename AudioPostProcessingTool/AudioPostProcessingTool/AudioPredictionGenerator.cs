using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Psi;
using Microsoft.Psi.Components;

namespace AudioPostProcessingTool
{ 

    class AudioPredictionGenerator : Generator
    {
        private System.IO.StreamReader reader;
        private bool mentionedEnd = false;
        private DateTime startDate;
        private DateTime pastDate;

        public Emitter<double> Pred { get; }

        public AudioPredictionGenerator(Pipeline p, string fileName, DateTime startDate, string name)
            : base(p)
        {
            this.Pred = p.CreateEmitter<double>(this, name);

            this.startDate = startDate;
            this.reader = new System.IO.StreamReader(File.OpenRead(fileName));

            Console.WriteLine($"Starting with params fileName: {fileName}, startDate: {startDate}, name: {name} ");
        }

        protected override DateTime GenerateNext(DateTime currentTime)
        {

            if (reader.EndOfStream)
            {
                if (!mentionedEnd)
                {
                    Console.WriteLine("Done!! :-)");
                    mentionedEnd = true;
                }

                return currentTime; // no more data
            }

            string line = this.reader.ReadLine();
            string[] values;


            if (!String.IsNullOrWhiteSpace(line))
            {
                values = line.Split(';');
            }
            else
            {
                Console.WriteLine("I am here?");
                return currentTime;
            }

            DateTime date = this.startDate.AddSeconds(Double.Parse(values[0]));

            if (date.Ticks < startDate.Ticks)
            {
                Console.WriteLine("Stopping: date.ticks was smaller then startDate.ticks");
                return new DateTime(0);
            }

            Console.WriteLine(date);


            var originatingTime = date;
            try
            {
                double val = Double.Parse(values[1]);
                Console.WriteLine($"Value: {values[1]} | Derived: {val}");
                this.Pred.Post(val, originatingTime);
                
            }
            catch (Exception e)
            {
                if (date.Ticks < pastDate.Ticks)
                {
                    throw new InvalidOperationException();
                }
            }

            pastDate = date;

            return originatingTime;
        }

    }
}
