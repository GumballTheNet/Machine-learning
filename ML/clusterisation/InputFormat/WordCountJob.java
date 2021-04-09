import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.fs.FileSystem;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class WordCountJob extends Configured implements Tool {
    public static class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        static final IntWritable one = new IntWritable(1);
        static final Pattern pattern = Pattern.compile("\\p{L}+");
        @Override
        protected void map(final LongWritable key, final Text value, final Context context)
                throws IOException, InterruptedException {
                Set<String> FullMatches = new HashSet<>();
                Matcher mat = pattern.matcher(value.toString().toLowerCase());
                while (mat.find()) {
                   FullMatches.add(mat.group());
                }
                for(String word: FullMatches) {
                   context.write(new Text(word), one);  
                } 
        }
    }

    public static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(final Text word, final Iterable<IntWritable> nums, final Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (final IntWritable i : nums) {
                sum += i.get();
            }
            context.write(word, new IntWritable(sum));
        }
    }

    private Job getJobConf(final String input, final String output) throws IOException {
        final Job job = Job.getInstance(getConf());
        job.setJarByClass(WordCountJob.class);
        job.setJobName(WordCountJob.class.getCanonicalName());
        job.setInputFormatClass(MyFileInputFormat.class);
        MyFileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.setMapperClass(WordCountMapper.class);
        job.setReducerClass(WordCountReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        return job;
    }

    @Override
    public int run(final String[] args) throws Exception {
        final Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(final String[] args) throws Exception {
        final int ret = ToolRunner.run(new WordCountJob(), args);
        System.exit(ret);
    }
}
