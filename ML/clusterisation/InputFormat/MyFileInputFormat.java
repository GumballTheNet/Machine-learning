import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;
import com.google.common.io.LittleEndianDataInputStream;

public class MyFileInputFormat extends FileInputFormat<LongWritable, Text> {
    public class MyRecordReader extends RecordReader<LongWritable, Text> {
        FSDataInputStream input_file;
        FSDataInputStream input_index;
        Text text;
        List<Integer> index;
        int max_size_buf;
        byte[] buf;
        byte[] decompressed;
        int current_record = 0;
        long split_len;
        long first_record;
	long split_beg;
	
        @Override
        public void initialize(final InputSplit split, final TaskAttemptContext context)
                throws IOException {
            final Configuration conf = context.getConfiguration();
            final FileSplit fsplit = (FileSplit) split;
            final Path path = fsplit.getPath(); 
            final String index_string = path.getName() + ".idx";
            final Path index_path = new Path(path.getParent(), index_string);
            final FileSystem fs = path.getFileSystem(conf);
            long offset = 0;
            input_index = fs.open(index_path);
            index = read_index(input_index);
            split_beg = fsplit.getStart();
            split_len = fsplit.getLength();
            while (current_record < split_beg) { 
                offset += index.get(current_record);
                ++current_record;
            }
            input_file = fs.open(path);
            input_file.seek(offset);
            decompressed = new byte[Collections.max(index) * 100];
           
        }

    
        @Override
        public boolean nextKeyValue() throws IOException {
            if (current_record - split_beg >= split_len) {
                return false;
            }
            buf = new byte[index.get(current_record)];
            input_file.readFully(buf, 0, index.get(current_record));
            final Inflater decompressor = new Inflater();
            decompressor.setInput(buf);
            final ByteArrayOutputStream stream = new ByteArrayOutputStream();
            int len = 1;
            while (len > 0){
            	try {
                   len = decompressor.inflate(decompressed);
                   stream.write(decompressed, 0, len);
                } catch (final DataFormatException e) {
                   len = 0;
                }

            }
            decompressor.end();
            stream.close();
            text = new Text(stream.toString(StandardCharsets.UTF_8.name())); 
            ++current_record;
            return true;
        }

        @Override
        public LongWritable getCurrentKey() throws IOException, InterruptedException {
            return new LongWritable((index.get(current_record - 1)));
        }

        @Override
        public Text getCurrentValue() throws IOException, InterruptedException {
            return text;
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            return (float) (current_record) / split_len;
        }

        @Override
        public void close() throws IOException {
            IOUtils.closeStream(input_file);
        }
    }

    public RecordReader<LongWritable, Text> createRecordReader(final InputSplit split, final TaskAttemptContext context)
            throws IOException, InterruptedException {
        final MyRecordReader reader = new MyRecordReader();
        reader.initialize(split, context);
        return reader;
    }

   private static List<Integer> read_index(final FSDataInputStream index_file) throws IOException {
        final LittleEndianDataInputStream in = new LittleEndianDataInputStream(index_file);
        final List<Integer> tmp = new ArrayList<>();
        try {
            while (true) {
                tmp.add(in.readInt());
            }
        } catch (final EOFException ex) {
        }
        return tmp;
    }

    public List<InputSplit> getSplits(final JobContext context) throws IOException {
        final List<InputSplit> splits = new ArrayList<>();

        for (final FileStatus status : listStatus(context)) {
            final Configuration conf = context.getConfiguration();
            final long min_split_size = conf.getLong("mapreduce.input.doc.bytes_per_map", 134217728);         
            final Path path = status.getPath();
            String index_string = path.getName();
            if (index_string.endsWith(".idx")) {
                continue;
            } else {
                index_string = index_string + ".idx";
            }
            final FileSystem fs = path.getFileSystem(conf);
            final Path index_path = new Path(path.getParent(), index_string);  
            final FSDataInputStream input_index = fs.open(index_path);
            final List<Integer> index = read_index(input_index);
            long cur_split = 0;   
            long current_split_size = 0; 
            long offset = 0;    
            for (final Integer cur : index) {
                current_split_size += cur;
                cur_split++;
                if (current_split_size > min_split_size) {
                    splits.add(new FileSplit(path, offset, cur_split, null));
                    offset += cur_split;
                    current_split_size = 0;
                    cur_split = 0;
                }
            }
            splits.add(new FileSplit(path, offset, cur_split, null));
        }
        return splits;
    }
}
